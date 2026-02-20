// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/dataset/file_orc.h"

#include <memory>
#include <optional>

#include "arrow/adapters/orc/adapter.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/file_base.h"
#include "arrow/dataset/scanner.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/util/thread_pool.h"

// ORC C++ library headers for type information
#include "orc/Type.hh"

namespace arrow {

using internal::checked_pointer_cast;

namespace dataset {

namespace {

// Helper function to build OrcSchemaField recursively
// ORC uses depth-first pre-order traversal: column 0 = root struct, 1+ = data columns
// column_index is passed by reference and incremented as we traverse
void BuildSchemaFieldRecursive(const std::shared_ptr<Field>& arrow_field,
                                const orc::Type* orc_type, int* column_index,
                                OrcSchemaField* out,
                                std::unordered_map<int, const OrcSchemaField*>* index_map,
                                std::unordered_map<const OrcSchemaField*, const OrcSchemaField*>* parent_map) {
  out->field = arrow_field;

  // Increment column index for this node
  ++(*column_index);
  int current_column = *column_index;

  // Determine if this is a leaf node based on ORC type
  // Leaves are primitive types that have statistics
  orc::TypeKind kind = orc_type->getKind();
  bool is_leaf = (kind != orc::STRUCT && kind != orc::LIST && kind != orc::MAP && kind != orc::UNION);

  if (is_leaf) {
    // Assign column index for leaf nodes (these have statistics)
    out->column_index = current_column;
    (*index_map)[current_column] = out;
  } else {
    // Container types: recursively process children
    out->column_index = -1;  // Containers don't have direct statistics

    // Get number of children
    uint64_t num_children = orc_type->getSubtypeCount();
    out->children.reserve(num_children);

    for (uint64_t i = 0; i < num_children; ++i) {
      OrcSchemaField child_field;
      const orc::Type* child_orc_type = orc_type->getSubtype(i);

      // For struct types, match Arrow field by name
      // For list/map types, use positional matching
      std::shared_ptr<Field> child_arrow_field;
      if (arrow_field->type()->id() == Type::STRUCT) {
        auto struct_type = std::static_pointer_cast<StructType>(arrow_field->type());
        child_arrow_field = struct_type->field(static_cast<int>(i));
      } else if (arrow_field->type()->id() == Type::LIST) {
        auto list_type = std::static_pointer_cast<ListType>(arrow_field->type());
        child_arrow_field = list_type->value_field();
      } else if (arrow_field->type()->id() == Type::MAP) {
        auto map_type = std::static_pointer_cast<MapType>(arrow_field->type());
        if (i == 0) {
          child_arrow_field = map_type->key_field();
        } else {
          child_arrow_field = map_type->item_field();
        }
      } else {
        // Fallback: create a dummy field
        child_arrow_field = field("child_" + std::to_string(i), null());
      }

      BuildSchemaFieldRecursive(child_arrow_field, child_orc_type, column_index,
                                 &child_field, index_map, parent_map);

      out->children.push_back(std::move(child_field));
      (*parent_map)[&out->children.back()] = out;
    }
  }
}

}  // namespace

// OrcSchemaManifest implementation
Status OrcSchemaManifest::Make(const std::shared_ptr<Schema>& schema,
                                const void* orc_type_ptr, OrcSchemaManifest* manifest) {
  if (!orc_type_ptr) {
    return Status::Invalid("ORC type pointer is null");
  }

  // Cast void* back to orc::Type*
  const orc::Type* orc_type = static_cast<const orc::Type*>(orc_type_ptr);

  // Validate that the root ORC type is a STRUCT
  if (orc_type->getKind() != orc::STRUCT) {
    return Status::Invalid("ORC root type must be STRUCT");
  }

  manifest->origin_schema = schema;
  manifest->schema_fields.clear();
  manifest->column_index_to_field.clear();
  manifest->child_to_parent.clear();

  // ORC column 0 is the root struct itself
  // User columns start at index 1
  int column_index = 0;  // Will be incremented to 1 for first field

  // Build schema fields for each top-level field
  uint64_t num_fields = orc_type->getSubtypeCount();
  manifest->schema_fields.reserve(num_fields);

  for (uint64_t i = 0; i < num_fields && i < static_cast<uint64_t>(schema->num_fields()); ++i) {
    OrcSchemaField field;
    const orc::Type* child_orc_type = orc_type->getSubtype(i);
    std::shared_ptr<Field> arrow_field = schema->field(static_cast<int>(i));

    BuildSchemaFieldRecursive(arrow_field, child_orc_type, &column_index, &field,
                               &manifest->column_index_to_field,
                               &manifest->child_to_parent);

    manifest->schema_fields.push_back(std::move(field));
  }

  return Status::OK();
}

// Helper function to resolve FieldRef to ORC column index using the manifest
// Returns std::nullopt if the field is not found or is not a leaf node
std::optional<int> GetOrcColumnIndex(const compute::FieldRef& field_ref,
                                      const OrcSchemaManifest& manifest) {
  // Try to resolve the FieldRef to a field in the schema
  auto maybe_match = field_ref.FindOne(*manifest.origin_schema);
  if (!maybe_match.ok()) {
    // Field not found in schema
    return std::nullopt;
  }

  const compute::FieldPath& field_path = *maybe_match;

  // Traverse the manifest to find the corresponding OrcSchemaField
  const OrcSchemaField* current_field = nullptr;

  // Start with top-level fields
  for (size_t i = 0; i < field_path.indices().size(); ++i) {
    int field_index = field_path.indices()[i];

    if (i == 0) {
      // Top-level field
      if (field_index < 0 || static_cast<size_t>(field_index) >= manifest.schema_fields.size()) {
        return std::nullopt;
      }
      current_field = &manifest.schema_fields[field_index];
    } else {
      // Nested field
      if (!current_field || field_index < 0 ||
          static_cast<size_t>(field_index) >= current_field->children.size()) {
        return std::nullopt;
      }
      current_field = &current_field->children[field_index];
    }
  }

  // Check if we found a field and if it's a leaf node
  if (current_field && current_field->is_leaf()) {
    return current_field->column_index;
  }

  // Not a leaf node or not found
  return std::nullopt;
}

namespace {

Result<std::unique_ptr<arrow::adapters::orc::ORCFileReader>> OpenORCReader(
    const FileSource& source,
    const std::shared_ptr<ScanOptions>& scan_options = nullptr) {
  ARROW_ASSIGN_OR_RAISE(auto input, source.Open());

  arrow::MemoryPool* pool;
  if (scan_options) {
    pool = scan_options->pool;
  } else {
    pool = default_memory_pool();
  }

  auto reader = arrow::adapters::orc::ORCFileReader::Open(std::move(input), pool);
  auto status = reader.status();
  if (!status.ok()) {
    return status.WithMessage("Could not open ORC input source '", source.path(),
                              "': ", status.message());
  }
  return reader;
}

/// \brief A ScanTask backed by an ORC file.
class OrcScanTask {
 public:
  OrcScanTask(std::shared_ptr<FileFragment> fragment,
              std::shared_ptr<ScanOptions> options)
      : fragment_(std::move(fragment)), options_(std::move(options)) {}

  Result<RecordBatchIterator> Execute() {
    struct Impl {
      static Result<RecordBatchIterator> Make(const FileSource& source,
                                              const FileFormat& format,
                                              const ScanOptions& scan_options) {
        ARROW_ASSIGN_OR_RAISE(
            auto reader,
            OpenORCReader(source, std::make_shared<ScanOptions>(scan_options)));

        auto materialized_fields = scan_options.MaterializedFields();
        // filter out virtual columns
        std::vector<std::string> included_fields;
        ARROW_ASSIGN_OR_RAISE(auto schema, reader->ReadSchema());
        for (const auto& ref : materialized_fields) {
          ARROW_ASSIGN_OR_RAISE(auto match, ref.FindOneOrNone(*schema));
          if (match.indices().empty()) continue;

          included_fields.push_back(schema->field(match.indices()[0])->name());
        }

        std::shared_ptr<RecordBatchReader> record_batch_reader;
        ARROW_ASSIGN_OR_RAISE(
            record_batch_reader,
            reader->GetRecordBatchReader(scan_options.batch_size, included_fields));

        return RecordBatchIterator(Impl{std::move(record_batch_reader)});
      }

      Result<std::shared_ptr<RecordBatch>> Next() {
        std::shared_ptr<RecordBatch> batch;
        RETURN_NOT_OK(record_batch_reader_->ReadNext(&batch));
        return batch;
      }

      std::shared_ptr<RecordBatchReader> record_batch_reader_;
    };

    return Impl::Make(fragment_->source(),
                      *checked_pointer_cast<FileFragment>(fragment_)->format(),
                      *options_);
  }

 private:
  std::shared_ptr<FileFragment> fragment_;
  std::shared_ptr<ScanOptions> options_;
};

class OrcScanTaskIterator {
 public:
  static Result<Iterator<std::shared_ptr<OrcScanTask>>> Make(
      std::shared_ptr<ScanOptions> options, std::shared_ptr<FileFragment> fragment) {
    return Iterator<std::shared_ptr<OrcScanTask>>(
        OrcScanTaskIterator(std::move(options), std::move(fragment)));
  }

  Result<std::shared_ptr<OrcScanTask>> Next() {
    if (once_) {
      // Iteration is done.
      return nullptr;
    }

    once_ = true;
    return std::make_shared<OrcScanTask>(fragment_, options_);
  }

 private:
  OrcScanTaskIterator(std::shared_ptr<ScanOptions> options,
                      std::shared_ptr<FileFragment> fragment)
      : options_(std::move(options)), fragment_(std::move(fragment)) {}

  bool once_ = false;
  std::shared_ptr<ScanOptions> options_;
  std::shared_ptr<FileFragment> fragment_;
};

}  // namespace

OrcFileFormat::OrcFileFormat() : FileFormat(/*default_fragment_scan_options=*/nullptr) {}

Result<bool> OrcFileFormat::IsSupported(const FileSource& source) const {
  RETURN_NOT_OK(source.Open().status());
  return OpenORCReader(source).ok();
}

Result<std::shared_ptr<Schema>> OrcFileFormat::Inspect(const FileSource& source) const {
  ARROW_ASSIGN_OR_RAISE(auto reader, OpenORCReader(source));
  return reader->ReadSchema();
}

Result<RecordBatchGenerator> OrcFileFormat::ScanBatchesAsync(
    const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<FileFragment>& file) const {
  // TODO investigate "true" async version
  // (https://issues.apache.org/jira/browse/ARROW-13795)
  ARROW_ASSIGN_OR_RAISE(auto task_iter, OrcScanTaskIterator::Make(options, file));
  struct IterState {
    Iterator<std::shared_ptr<OrcScanTask>> iter;
    RecordBatchIterator curr_iter;
    bool first;
    ::arrow::internal::Executor* io_executor;
  };
  struct {
    Future<std::shared_ptr<RecordBatch>> operator()() {
      auto state = state_;
      return ::arrow::DeferNotOk(
          state->io_executor->Submit([state]() -> Result<std::shared_ptr<RecordBatch>> {
            if (state->first) {
              ARROW_ASSIGN_OR_RAISE(auto task, state->iter.Next());
              ARROW_ASSIGN_OR_RAISE(state->curr_iter, task->Execute());
              state->first = false;
            }
            while (!IsIterationEnd(state->curr_iter)) {
              ARROW_ASSIGN_OR_RAISE(auto next_batch, state->curr_iter.Next());
              if (IsIterationEnd(next_batch)) {
                ARROW_ASSIGN_OR_RAISE(auto task, state->iter.Next());
                if (IsIterationEnd(task)) {
                  state->curr_iter = IterationEnd<RecordBatchIterator>();
                } else {
                  ARROW_ASSIGN_OR_RAISE(state->curr_iter, task->Execute());
                }
              } else {
                return next_batch;
              }
            }
            return IterationEnd<std::shared_ptr<RecordBatch>>();
          }));
    }
    std::shared_ptr<IterState> state_;
  } iter_to_gen{std::shared_ptr<IterState>(
      new IterState{std::move(task_iter), {}, true, options->io_context.executor()})};
  return iter_to_gen;
}

Future<std::optional<int64_t>> OrcFileFormat::CountRows(
    const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
    const std::shared_ptr<ScanOptions>& options) {
  if (ExpressionHasFieldRefs(predicate)) {
    return Future<std::optional<int64_t>>::MakeFinished(std::nullopt);
  }
  auto self = checked_pointer_cast<OrcFileFormat>(shared_from_this());
  return DeferNotOk(options->io_context.executor()->Submit(
      [self, file]() -> Result<std::optional<int64_t>> {
        ARROW_ASSIGN_OR_RAISE(auto reader, OpenORCReader(file->source()));
        return reader->NumberOfRows();
      }));
}

// //
// // OrcFileWriter, OrcFileWriteOptions
// //

std::shared_ptr<FileWriteOptions> OrcFileFormat::DefaultWriteOptions() {
  // TODO (https://issues.apache.org/jira/browse/ARROW-13796)
  return nullptr;
}

Result<std::shared_ptr<FileWriter>> OrcFileFormat::MakeWriter(
    std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
    std::shared_ptr<FileWriteOptions> options,
    fs::FileLocator destination_locator) const {
  // TODO (https://issues.apache.org/jira/browse/ARROW-13796)
  return Status::NotImplemented("ORC writer not yet implemented.");
}

}  // namespace dataset
}  // namespace arrow
