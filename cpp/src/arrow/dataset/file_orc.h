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
#include <optional>
#include <string>
#include <vector>

#include "arrow/dataset/file_base.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/io/type_fwd.h"
#include "arrow/result.h"

namespace arrow {

namespace adapters {
namespace orc {
struct OrcColumnStatisticsAsScalars;
struct OrcSchemaManifest;
class Statistics;
class ORCFileReader;
}  // namespace orc
}  // namespace adapters

namespace dataset {

/// \addtogroup dataset-file-formats
///
/// @{

// Forward declaration
class OrcFileFragment;

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

  using FileFormat::MakeFragment;

  /// \brief Create a Fragment targeting all stripes.
  Result<std::shared_ptr<FileFragment>> MakeFragment(
      FileSource source, compute::Expression partition_expression,
      std::shared_ptr<Schema> physical_schema) override;

  /// \brief Create a Fragment, restricted to the specified stripes.
  Result<std::shared_ptr<OrcFileFragment>> MakeFragment(
      FileSource source, compute::Expression partition_expression,
      std::shared_ptr<Schema> physical_schema, std::vector<int64_t> stripes);
};

/// \brief A FileFragment with ORC logic.
///
/// OrcFileFragment provides a lazy (with respect to IO) interface to
/// scan ORC files with predicate pushdown. Metadata is cached on first access.
class ARROW_DS_EXPORT OrcFileFragment : public FileFragment {
 public:
  /// \brief Return the stripes selected by this fragment.
  const std::vector<int64_t>& stripes() const {
    if (stripes_) return *stripes_;
    static std::vector<int64_t> empty;
    return empty;
  }

  /// \brief Ensure this fragment's metadata is in memory.
  Status EnsureCompleteMetadata(adapters::orc::ORCFileReader* reader = nullptr);

  /// \brief Evaluate column statistics as an Arrow expression.
  ///
  /// Given a field and its statistics, build a filtering expression that
  /// represents the min/max bounds. For example, if min=10 and max=20,
  /// this returns (field >= 10) AND (field <= 20).
  ///
  /// \param field the Arrow field
  /// \param field_ref the field reference in the expression
  /// \param statistics the ORC column statistics
  /// \return expression representing statistics bounds, or nullopt if not available
  static std::optional<compute::Expression> EvaluateStatisticsAsExpression(
      const Field& field, const FieldRef& field_ref,
      const adapters::orc::Statistics& statistics);
  static std::optional<compute::Expression> EvaluateStatisticsAsExpression(
      const Field& field, const FieldRef& field_ref,
      const adapters::orc::OrcColumnStatisticsAsScalars& statistics);

 private:
  OrcFileFragment(FileSource source, std::shared_ptr<FileFormat> format,
                  compute::Expression partition_expression,
                  std::shared_ptr<Schema> physical_schema,
                  std::optional<std::vector<int64_t>> stripes);

  Status SetMetadata(std::shared_ptr<adapters::orc::OrcSchemaManifest> manifest);

  /// Return a filtered subset of stripe indices.
  Result<std::vector<int64_t>> FilterStripes(compute::Expression predicate);

  /// Simplify the predicate against the statistics of each stripe.
  Result<std::vector<compute::Expression>> TestStripes(compute::Expression predicate);

  /// Try to count rows matching the predicate using metadata. Expects
  /// metadata to be present, and expects the predicate to have been
  /// simplified against the partition expression already.
  Result<std::optional<int64_t>> TryCountRows(compute::Expression predicate);

  OrcFileFormat& orc_format_;

  /// Indices of stripes selected by this fragment,
  /// or std::nullopt if all stripes are selected.
  std::optional<std::vector<int64_t>> stripes_;

  /// The expressions (combined for all columns for which statistics have been
  /// processed) are stored per stripe
  std::vector<compute::Expression> statistics_expressions_;

  /// Statistics status are kept track of per stripe
  std::vector<bool> statistics_expressions_complete_;

  /// Cached ORC schema manifest
  std::shared_ptr<adapters::orc::OrcSchemaManifest> manifest_;

  /// Total number of stripes (cached after first metadata load)
  int64_t num_stripes_ = -1;

  friend class OrcFileFormat;
};

/// @}

}  // namespace dataset
}  // namespace arrow
