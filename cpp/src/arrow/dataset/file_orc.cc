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

#include <cmath>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "arrow/adapters/orc/adapter.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/file_base.h"
#include "arrow/dataset/scanner.h"
#include "arrow/scalar.h"
#include "arrow/table.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/thread_pool.h"

namespace arrow {
namespace adapters {
namespace orc {

// Struct definitions duplicated from util.h to avoid pulling in orc/OrcFile.hh.
// These are pure Arrow types with no liborc dependency.
struct OrcSchemaField {
  std::shared_ptr<Field> field;
  int orc_column_id;
  std::vector<OrcSchemaField> children;
  bool is_leaf() const { return children.empty(); }
};

struct OrcSchemaManifest {
  std::vector<OrcSchemaField> schema_fields;
  const OrcSchemaField* GetField(const std::vector<int>& path) const;
};

}  // namespace orc
}  // namespace adapters

using internal::checked_cast;
using internal::checked_pointer_cast;

namespace dataset {

namespace {

inline void FoldingAnd(compute::Expression* l, compute::Expression r) {
  if (*l == compute::literal(true)) {
    *l = std::move(r);
  } else {
    *l = compute::and_(std::move(*l), std::move(r));
  }
}

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
  auto orc_fragment = checked_pointer_cast<OrcFileFragment>(file);

  // State for the async generator that performs stripe filtering and reading
  struct State {
    std::shared_ptr<ScanOptions> options;
    std::shared_ptr<OrcFileFragment> orc_fragment;
    RecordBatchIterator batch_iterator;
    bool initialized = false;

    Result<std::shared_ptr<RecordBatch>> Next() {
      if (!initialized) {
        ARROW_ASSIGN_OR_RAISE(auto reader,
                              OpenORCReader(orc_fragment->source(), options));

        RETURN_NOT_OK(orc_fragment->EnsureCompleteMetadata(reader.get()));

        ARROW_ASSIGN_OR_RAISE(auto filtered_stripes,
                              orc_fragment->FilterStripes(options->filter));

        if (filtered_stripes.empty()) {
          batch_iterator = MakeEmptyIterator<std::shared_ptr<RecordBatch>>();
          initialized = true;
          return IterationEnd<std::shared_ptr<RecordBatch>>();
        }

        // Build column projection using field names (same approach as OrcScanTask)
        auto materialized_fields = options->MaterializedFields();
        std::vector<std::string> included_fields;
        ARROW_ASSIGN_OR_RAISE(auto schema, reader->ReadSchema());
        for (const auto& ref : materialized_fields) {
          ARROW_ASSIGN_OR_RAISE(auto match, ref.FindOneOrNone(*schema));
          if (match.indices().empty()) continue;
          included_fields.push_back(schema->field(match.indices()[0])->name());
        }

        // Read filtered stripes with column projection
        std::vector<std::shared_ptr<RecordBatch>> stripe_batches;
        stripe_batches.reserve(filtered_stripes.size());
        for (int64_t stripe_idx : filtered_stripes) {
          if (included_fields.empty()) {
            ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadStripe(stripe_idx));
            stripe_batches.push_back(std::move(batch));
          } else {
            ARROW_ASSIGN_OR_RAISE(auto batch,
                                  reader->ReadStripe(stripe_idx, included_fields));
            stripe_batches.push_back(std::move(batch));
          }
        }
        ARROW_ASSIGN_OR_RAISE(auto table,
                              Table::FromRecordBatches(std::move(stripe_batches)));

        auto batch_reader = std::make_shared<TableBatchReader>(table);
        batch_reader->set_chunksize(options->batch_size);

        auto iter_fn = [batch_reader]() -> Result<std::shared_ptr<RecordBatch>> {
          std::shared_ptr<RecordBatch> batch;
          RETURN_NOT_OK(batch_reader->ReadNext(&batch));
          if (batch == nullptr) {
            return IterationEnd<std::shared_ptr<RecordBatch>>();
          }
          return batch;
        };

        batch_iterator = MakeFunctionIterator(iter_fn);
        initialized = true;
      }

      return batch_iterator.Next();
    }
  };

  auto state = std::make_shared<State>();
  state->options = options;
  state->orc_fragment = orc_fragment;

  auto generator = [state]() -> Future<std::shared_ptr<RecordBatch>> {
    return DeferNotOk(state->options->io_context.executor()->Submit(
        [state]() -> Result<std::shared_ptr<RecordBatch>> {
          return state->Next();
        }));
  };

  return generator;
}

Future<std::optional<int64_t>> OrcFileFormat::CountRows(
    const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
    const std::shared_ptr<ScanOptions>& options) {
  auto orc_file = checked_pointer_cast<OrcFileFragment>(file);

  return DeferNotOk(options->io_context.executor()->Submit(
      [orc_file, predicate]() -> Result<std::optional<int64_t>> {
        RETURN_NOT_OK(orc_file->EnsureCompleteMetadata());
        return orc_file->TryCountRows(predicate);
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

Result<std::shared_ptr<FileFragment>> OrcFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema) {
  return std::shared_ptr<FileFragment>(
      new OrcFileFragment(std::move(source), shared_from_this(),
                          std::move(partition_expression), std::move(physical_schema),
                          std::nullopt));
}

Result<std::shared_ptr<OrcFileFragment>> OrcFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema, std::vector<int64_t> stripes) {
  return std::shared_ptr<OrcFileFragment>(
      new OrcFileFragment(std::move(source), shared_from_this(),
                          std::move(partition_expression), std::move(physical_schema),
                          std::move(stripes)));
}

//
// OrcFileFragment
//

OrcFileFragment::OrcFileFragment(FileSource source,
                                 std::shared_ptr<FileFormat> format,
                                 compute::Expression partition_expression,
                                 std::shared_ptr<Schema> physical_schema,
                                 std::optional<std::vector<int64_t>> stripes)
    : FileFragment(std::move(source), std::move(format),
                   std::move(partition_expression), std::move(physical_schema)),
      orc_format_(checked_cast<OrcFileFormat&>(*format_)),
      stripes_(std::move(stripes)) {}

Status OrcFileFragment::SetMetadata(
    std::shared_ptr<adapters::orc::OrcSchemaManifest> manifest) {
  manifest_ = std::move(manifest);
  return Status::OK();
}

Status OrcFileFragment::EnsureCompleteMetadata(
    adapters::orc::ORCFileReader* reader) {
  std::unique_ptr<adapters::orc::ORCFileReader> reader_holder;
  if (reader == nullptr) {
    ARROW_ASSIGN_OR_RAISE(reader_holder, OpenORCReader(source_));
    reader = reader_holder.get();
  }

  // Build the manifest if not yet cached
  if (!manifest_) {
    ARROW_ASSIGN_OR_RAISE(auto schema, reader->ReadSchema());
    ARROW_ASSIGN_OR_RAISE(manifest_,
                          reader->BuildSchemaManifest(schema));
    // Set physical_schema_ if not already set (fragment may have been created with
    // nullptr schema). This avoids a null deref in ReadPhysicalSchemaImpl().
    if (physical_schema_ == nullptr) {
      physical_schema_ = std::move(schema);
    }
  }

  // Cache stripe count and initialize statistics structures
  if (num_stripes_ < 0) {
    num_stripes_ = reader->NumberOfStripes();
    if (!stripes_) {
      // Initialize statistics structures for all stripes
      statistics_expressions_.resize(num_stripes_, compute::literal(true));
      statistics_expressions_complete_.resize(num_stripes_, false);
    } else {
      // Initialize statistics structures for selected stripes
      statistics_expressions_.resize(stripes_->size(), compute::literal(true));
      statistics_expressions_complete_.resize(stripes_->size(), false);
    }
  }

  return Status::OK();
}

std::optional<compute::Expression> OrcFileFragment::EvaluateStatisticsAsExpression(
    const Field& field, const FieldRef& field_ref,
    const adapters::orc::OrcColumnStatistics& statistics) {
  if (!statistics.has_min_max) {
    return std::nullopt;
  }

  // Both min and max must be valid
  if (!statistics.min || !statistics.max) {
    return std::nullopt;
  }

  // Check for float NaN: if both are NaN, no valid bounds
  if (is_floating(field.type()->id())) {
    bool min_is_nan = false;
    bool max_is_nan = false;

    if (statistics.min->is_valid) {
      switch (field.type()->id()) {
        case Type::FLOAT: {
          auto val = checked_cast<const FloatScalar&>(*statistics.min).value;
          min_is_nan = std::isnan(val);
          break;
        }
        case Type::DOUBLE: {
          auto val = checked_cast<const DoubleScalar&>(*statistics.min).value;
          min_is_nan = std::isnan(val);
          break;
        }
        default:
          break;
      }
    }

    if (statistics.max->is_valid) {
      switch (field.type()->id()) {
        case Type::FLOAT: {
          auto val = checked_cast<const FloatScalar&>(*statistics.max).value;
          max_is_nan = std::isnan(val);
          break;
        }
        case Type::DOUBLE: {
          auto val = checked_cast<const DoubleScalar&>(*statistics.max).value;
          max_is_nan = std::isnan(val);
          break;
        }
        default:
          break;
      }
    }

    if (min_is_nan && max_is_nan) {
      return std::nullopt;
    }
  }

  // Build the bounds expression: (field >= min) AND (field <= max)
  compute::Expression field_expr = compute::field_ref(field_ref);
  compute::Expression min_expr = compute::literal(true);
  compute::Expression max_expr = compute::literal(true);

  if (statistics.min->is_valid) {
    min_expr = compute::greater_equal(field_expr, compute::literal(statistics.min));
  }

  if (statistics.max->is_valid) {
    max_expr = compute::less_equal(field_expr, compute::literal(statistics.max));
  }

  auto bounds_expr = compute::and_(min_expr, max_expr);

  // If the column has nulls, include is_null in the expression
  if (statistics.has_null) {
    return compute::or_(compute::is_null(field_expr), bounds_expr);
  }

  return bounds_expr;
}

Result<std::vector<int64_t>> OrcFileFragment::FilterStripes(
    compute::Expression predicate) {
  std::vector<int64_t> filtered_stripes;
  ARROW_ASSIGN_OR_RAISE(auto expressions, TestStripes(std::move(predicate)));

  // Determine the stripe indices
  std::vector<int64_t> stripe_indices;
  if (stripes_) {
    stripe_indices = *stripes_;
  } else {
    stripe_indices.resize(num_stripes_);
    for (int64_t i = 0; i < num_stripes_; ++i) {
      stripe_indices[i] = i;
    }
  }

  DCHECK(expressions.empty() || (expressions.size() == stripe_indices.size()));

  for (size_t i = 0; i < expressions.size(); ++i) {
    if (expressions[i].IsSatisfiable()) {
      filtered_stripes.push_back(stripe_indices[i]);
    }
  }

  return filtered_stripes;
}

Result<std::vector<compute::Expression>> OrcFileFragment::TestStripes(
    compute::Expression predicate) {
  ARROW_RETURN_NOT_OK(EnsureCompleteMetadata());

  ARROW_ASSIGN_OR_RAISE(
      predicate, SimplifyWithGuarantee(std::move(predicate), partition_expression_));

  if (!predicate.IsSatisfiable()) {
    return std::vector<compute::Expression>{};
  }

  // Determine the stripe indices to test
  std::vector<int64_t> stripe_indices;
  if (stripes_) {
    stripe_indices = *stripes_;
  } else {
    // All stripes
    stripe_indices.resize(num_stripes_);
    for (int64_t i = 0; i < num_stripes_; ++i) {
      stripe_indices[i] = i;
    }
  }

  // Open reader for statistics
  ARROW_ASSIGN_OR_RAISE(auto reader, OpenORCReader(source_));

  // Track which columns have been processed (dedup across repeated calls)
  std::unordered_set<int> processed_columns;

  // For each FieldRef in the predicate, update statistics expressions
  for (const FieldRef& ref : FieldsInExpression(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto match, ref.FindOneOrNone(*physical_schema_));

    if (match.empty()) continue;

    // Navigate to the OrcSchemaField via the manifest
    const adapters::orc::OrcSchemaField* schema_field =
        &manifest_->schema_fields[match[0]];

    for (size_t i = 1; i < match.indices().size(); ++i) {
      if (schema_field->field->type()->id() != Type::STRUCT) {
        return Status::Invalid("nested paths only supported for structs");
      }
      schema_field = &schema_field->children[match[i]];
    }

    // Only process leaf fields
    if (!schema_field->is_leaf()) continue;

    // Skip if already processed in this call
    int orc_column_id = schema_field->orc_column_id;
    if (!processed_columns.insert(orc_column_id).second) continue;

    // For each stripe, get statistics and build expression
    for (size_t i = 0; i < stripe_indices.size(); ++i) {
      int64_t stripe_idx = stripe_indices[i];

      ARROW_ASSIGN_OR_RAISE(
          auto stats, reader->GetStripeStatistics(stripe_idx, {orc_column_id}));

      if (!stats.empty()) {
        const auto& column_stats = stats[0];
        if (auto minmax =
                EvaluateStatisticsAsExpression(*schema_field->field, ref, column_stats)) {
          FoldingAnd(&statistics_expressions_[i], std::move(*minmax));
          ARROW_ASSIGN_OR_RAISE(statistics_expressions_[i],
                                statistics_expressions_[i].Bind(*physical_schema_));
        }
      }
    }
  }

  // Simplify predicate with each stripe's statistics
  std::vector<compute::Expression> stripe_expressions(stripe_indices.size());
  for (size_t i = 0; i < stripe_indices.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto stripe_predicate,
                          SimplifyWithGuarantee(predicate, statistics_expressions_[i]));
    stripe_expressions[i] = std::move(stripe_predicate);
  }

  return stripe_expressions;
}

Result<std::optional<int64_t>> OrcFileFragment::TryCountRows(
    compute::Expression predicate) {
  ARROW_RETURN_NOT_OK(EnsureCompleteMetadata());

  if (ExpressionHasFieldRefs(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto expressions, TestStripes(std::move(predicate)));

    // Determine the stripe indices
    std::vector<int64_t> stripe_indices;
    if (stripes_) {
      stripe_indices = *stripes_;
    } else {
      stripe_indices.resize(num_stripes_);
      for (int64_t i = 0; i < num_stripes_; ++i) {
        stripe_indices[i] = i;
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto reader, OpenORCReader(source_));

    int64_t rows = 0;
    for (size_t i = 0; i < stripe_indices.size(); ++i) {
      // If the stripe is entirely excluded, exclude it from the row count
      if (!expressions[i].IsSatisfiable()) continue;

      // Unless the stripe is entirely included, bail out of fast path
      if (expressions[i] != compute::literal(true)) return std::nullopt;

      auto stripe_info = reader->GetStripeInformation(stripe_indices[i]);
      rows += stripe_info.num_rows;
    }

    return rows;
  }

  // No field refs - return total row count
  ARROW_ASSIGN_OR_RAISE(auto reader, OpenORCReader(source_));
  return reader->NumberOfRows();
}

}  // namespace dataset
}  // namespace arrow
