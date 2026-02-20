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

// This API is EXPERIMENTAL.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/dataset/file_base.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/io/type_fwd.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace dataset {

/// \addtogroup dataset-file-formats
///
/// @{

/// \brief Bridge between an arrow::Field and ORC column indices.
///
/// Similar to Parquet's SchemaField, this structure maps Arrow schema fields
/// to ORC physical column indices. ORC uses a depth-first pre-order traversal
/// where column 0 is the root struct, and subsequent indices are for child columns.
struct ARROW_DS_EXPORT OrcSchemaField {
  /// The Arrow field corresponding to this ORC column
  std::shared_ptr<Field> field;

  /// Child fields (for nested types like structs, lists, maps)
  std::vector<OrcSchemaField> children;

  /// ORC column index (only set for leaf nodes that have statistics)
  /// For ORC, column 0 is the root struct, columns 1+ are the actual data columns
  int column_index = -1;

  /// Check if this is a leaf node (has column statistics)
  bool is_leaf() const { return column_index != -1; }
};

/// \brief Bridge between an ORC file schema and an Arrow Schema.
///
/// Maps Arrow schema fields to ORC physical column indices for statistics lookup.
/// Similar to Parquet's SchemaManifest but adapted for ORC's type system.
struct ARROW_DS_EXPORT OrcSchemaManifest {
  /// Create a schema manifest from ORC type information
  /// \param schema The Arrow schema
  /// \param orc_type Pointer to orc::Type from the ORC reader (as void* to avoid ORC header dependency)
  /// \param manifest Output manifest to populate
  static Status Make(const std::shared_ptr<Schema>& schema, const void* orc_type,
                     OrcSchemaManifest* manifest);

  /// The Arrow schema
  std::shared_ptr<Schema> origin_schema;

  /// Top-level schema fields
  std::vector<OrcSchemaField> schema_fields;

  /// Map from ORC column index to schema field (for fast lookup)
  std::unordered_map<int, const OrcSchemaField*> column_index_to_field;

  /// Map from child field to parent field (for traversal)
  std::unordered_map<const OrcSchemaField*, const OrcSchemaField*> child_to_parent;

  /// Get the schema field for a given ORC column index
  Status GetColumnField(int column_index, const OrcSchemaField** out) const {
    auto it = column_index_to_field.find(column_index);
    if (it == column_index_to_field.end()) {
      return Status::KeyError("Column index ", column_index,
                             " not found in ORC schema manifest");
    }
    *out = it->second;
    return Status::OK();
  }

  /// Get the parent field of a given field
  const OrcSchemaField* GetParent(const OrcSchemaField* field) const {
    auto it = child_to_parent.find(field);
    if (it == child_to_parent.end()) {
      return nullptr;
    }
    return it->second;
  }
};

constexpr char kOrcTypeName[] = "orc";

/// \brief A FileFormat implementation that reads from and writes to ORC files
class ARROW_DS_EXPORT OrcFileFormat : public FileFormat {
 public:
  OrcFileFormat();

  std::string type_name() const override { return kOrcTypeName; }

  bool Equals(const FileFormat& other) const override {
    return type_name() == other.type_name();
  }

  Result<bool> IsSupported(const FileSource& source) const override;

  /// \brief Return the schema of the file if possible.
  Result<std::shared_ptr<Schema>> Inspect(const FileSource& source) const override;

  Result<RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<ScanOptions>& options,
      const std::shared_ptr<FileFragment>& file) const override;

  Future<std::optional<int64_t>> CountRows(
      const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
      const std::shared_ptr<ScanOptions>& options) override;

  Result<std::shared_ptr<FileWriter>> MakeWriter(
      std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
      std::shared_ptr<FileWriteOptions> options,
      fs::FileLocator destination_locator) const override;

  std::shared_ptr<FileWriteOptions> DefaultWriteOptions() override;
};

/// @}

}  // namespace dataset
}  // namespace arrow
