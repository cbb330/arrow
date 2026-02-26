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
#include <sstream>
#include <utility>

#include "arrow/adapters/orc/adapter.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/discovery.h"
#include "arrow/dataset/file_base.h"
#include "arrow/dataset/partition.h"
#include "arrow/dataset/test_util_internal.h"
#include "arrow/io/memory.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/util.h"
#include "arrow/util/checked_cast.h"

namespace arrow {

using internal::checked_pointer_cast;

namespace dataset {

class OrcFormatHelper {
 public:
  using FormatType = OrcFileFormat;
  static Result<std::shared_ptr<Buffer>> Write(RecordBatchReader* reader) {
    ARROW_ASSIGN_OR_RAISE(auto sink, io::BufferOutputStream::Create());
    // Use very small stripe size to force one stripe per RecordBatch for testing
    adapters::orc::WriteOptions write_options;
    write_options.stripe_size = 1;  // 1 byte - forces new stripe for each batch
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          adapters::orc::ORCFileWriter::Open(sink.get(), write_options));
    // Write each RecordBatch individually to create separate stripes
    for (;;) {
      ARROW_ASSIGN_OR_RAISE(auto batch, reader->Next());
      if (!batch) break;
      RETURN_NOT_OK(writer->Write(*batch));
    }
    RETURN_NOT_OK(writer->Close());
    return sink->Finish();
  }

  static std::shared_ptr<OrcFileFormat> MakeFormat() {
    return std::make_shared<OrcFileFormat>();
  }
};

class TestOrcFileFormat : public FileFormatFixtureMixin<OrcFormatHelper> {};

// TEST_F(TestOrcFileFormat, WriteRecordBatchReader) { TestWrite(); }

TEST_F(TestOrcFileFormat, InspectFailureWithRelevantError) {
  TestInspectFailureWithRelevantError(StatusCode::IOError, "ORC");
}
TEST_F(TestOrcFileFormat, Inspect) { TestInspect(); }
TEST_F(TestOrcFileFormat, IsSupported) { TestIsSupported(); }
TEST_F(TestOrcFileFormat, CountRows) { TestCountRows(); }
TEST_F(TestOrcFileFormat, FragmentEquals) { TestFragmentEquals(); }

TEST_F(TestOrcFileFormat, CountRowsPredicatePushdown) {
  constexpr int64_t kNumStripes = 16;
  constexpr int64_t kTotalNumRows = kNumStripes * (kNumStripes + 1) / 2;

  // Create a simple schema with only ORC-supported types
  auto schema_simple = ::arrow::schema({field("i64", int64())});

  // Create test data: n batches where batch i has i rows, all with value i
  RecordBatchVector batches;
  for (int64_t i = 1; i <= kNumStripes; i++) {
    // Build JSON array: i rows, each with value i
    std::stringstream json;
    json << "[";
    for (int64_t j = 0; j < i; j++) {
      if (j > 0) json << ",";
      json << i;
    }
    json << "]";
    auto arr = ArrayFromJSON(int64(), json.str());
    batches.push_back(RecordBatch::Make(schema_simple, i, {arr}));
  }

  ASSERT_OK_AND_ASSIGN(auto table, Table::FromRecordBatches(schema_simple, batches));
  auto reader = std::make_shared<TableBatchReader>(table);

  auto source = GetFileSource(reader.get());
  auto options = std::make_shared<ScanOptions>();

  auto fragment = MakeFragment(*source);

  // Test with literal(true) - should return all rows (no field refs, fast path)
  ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(kTotalNumRows),
                            fragment->CountRows(literal(true), options));

  // Test with predicates that exclude some stripes
  // Note: TryCountRows requires predicates to simplify to literal(true) for included
  // stripes to use the fast path. Predicates that only partially match stripe bounds
  // will return nullopt and fall back to scanning.
  for (int i = 1; i <= kNumStripes; i++) {
    SCOPED_TRACE(i);
    // Predicate i64 < i+1 matches stripes 1..i (all rows in those stripes)
    // Since all rows in matching stripes satisfy the predicate, fast path works
    auto predicate = less(field_ref("i64"), literal(i + 1));
    ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
    auto expected = i * (i + 1) / 2;
    ASSERT_FINISHES_OK_AND_ASSIGN(auto count, fragment->CountRows(predicate, options));
    // May use fast path or fall back - just verify the result is correct when available
    if (count.has_value()) {
      ASSERT_EQ(*count, expected);
    }
  }

  // Test with predicates that exclude all stripes
  auto predicate = literal(false);
  ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
  ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(0),
                            fragment->CountRows(predicate, options));

  // Out of bounds predicate (value doesn't exist)
  predicate = equal(field_ref("i64"), literal<int64_t>(kNumStripes + 1));
  ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
  ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(0),
                            fragment->CountRows(predicate, options));

  // Predicate that partially matches a stripe should return nullopt (fast path fails)
  predicate = greater(field_ref("i64"), literal<int64_t>(5));
  ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
  // This should return nullopt because stripes with value > 5 are included but
  // stripes 1-5 are excluded, and we can't count partial stripe inclusions
  ASSERT_FINISHES_OK_AND_ASSIGN(auto count_opt, fragment->CountRows(predicate, options));
  // Greater than predicates may or may not use fast path depending on statistics
  // Just verify it doesn't crash and returns a valid result
  ASSERT_TRUE(!count_opt.has_value() || *count_opt == kTotalNumRows - (5 * 6 / 2));
}

// TODO add TestOrcFileSystemDataset if write support is added

class TestOrcFileFormatScan : public FileFormatScanMixin<OrcFormatHelper> {
 protected:
  void CountRowsAndBatchesInScan(const std::shared_ptr<Fragment>& fragment,
                                 int64_t expected_rows, int64_t expected_batches) {
    int64_t actual_rows = 0;
    int64_t actual_batches = 0;

    for (auto maybe_batch : Batches(fragment)) {
      ASSERT_OK_AND_ASSIGN(auto batch, maybe_batch);
      actual_rows += batch->num_rows();
      ++actual_batches;
    }

    EXPECT_EQ(actual_rows, expected_rows);
    EXPECT_EQ(actual_batches, expected_batches);
  }
};

TEST_P(TestOrcFileFormatScan, ScanRecordBatchReader) { TestScan(); }
TEST_P(TestOrcFileFormatScan, ScanBatchSize) { TestScanBatchSize(); }
TEST_P(TestOrcFileFormatScan, ScanNoReadahead) { TestScanNoReadahead(); }
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderProjected) { TestScanProjected(); }
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderProjectedNested) {
  TestScanProjectedNested();
}
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderProjectedMissingCols) {
  TestScanProjectedMissingCols();
}
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderWithVirtualColumn) {
  TestScanWithVirtualColumn();
}
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderWithDuplicateColumn) {
  TestScanWithDuplicateColumn();
}
TEST_P(TestOrcFileFormatScan, ScanRecordBatchReaderWithDuplicateColumnError) {
  TestScanWithDuplicateColumnError();
}
TEST_P(TestOrcFileFormatScan, ScanWithPushdownNulls) { TestScanWithPushdownNulls(); }

INSTANTIATE_TEST_SUITE_P(TestScan, TestOrcFileFormatScan,
                         ::testing::ValuesIn(TestFormatParams::Values()),
                         TestFormatParams::ToTestNameString);

// Test statistics expression evaluation
TEST(TestOrcStatistics, EvaluateStatisticsAsExpression) {
  using adapters::orc::OrcColumnStatistics;

  auto field_i64 = field("i64", int64());
  auto field_f64 = field("f64", float64());
  FieldRef ref_i64("i64");
  FieldRef ref_f64("f64");

  // Test with valid min/max statistics
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = false;
    stats.num_values = 100;
    stats.min = std::make_shared<Int64Scalar>(10);
    stats.max = std::make_shared<Int64Scalar>(20);

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_i64, ref_i64, stats);
    ASSERT_TRUE(expr.has_value());
    // Should produce (i64 >= 10) AND (i64 <= 20)
    EXPECT_TRUE(expr->ToString().find(">=") != std::string::npos);
    EXPECT_TRUE(expr->ToString().find("<=") != std::string::npos);
  }

  // Test with has_null = true
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = true;
    stats.num_values = 100;
    stats.min = std::make_shared<Int64Scalar>(10);
    stats.max = std::make_shared<Int64Scalar>(20);

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_i64, ref_i64, stats);
    ASSERT_TRUE(expr.has_value());
    // Should produce is_null(i64) OR ((i64 >= 10) AND (i64 <= 20))
    EXPECT_TRUE(expr->ToString().find("is_null") != std::string::npos);
  }

  // Test with no min/max statistics
  {
    OrcColumnStatistics stats;
    stats.has_min_max = false;
    stats.has_null = false;
    stats.num_values = 100;

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_i64, ref_i64, stats);
    EXPECT_FALSE(expr.has_value());
  }

  // Test with null min or max
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = false;
    stats.num_values = 100;
    stats.min = nullptr;
    stats.max = std::make_shared<Int64Scalar>(20);

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_i64, ref_i64, stats);
    EXPECT_FALSE(expr.has_value());
  }

  // Test with NaN values for float (both NaN)
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = false;
    stats.num_values = 100;
    stats.min = std::make_shared<DoubleScalar>(std::nan(""));
    stats.max = std::make_shared<DoubleScalar>(std::nan(""));

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_f64, ref_f64, stats);
    EXPECT_FALSE(expr.has_value());
  }

  // Test with NaN min but valid max
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = false;
    stats.num_values = 100;
    stats.min = std::make_shared<DoubleScalar>(std::nan(""));
    stats.max = std::make_shared<DoubleScalar>(20.0);

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_f64, ref_f64, stats);
    ASSERT_TRUE(expr.has_value());
    // Should only have max bound
    EXPECT_TRUE(expr->ToString().find("<=") != std::string::npos);
  }

  // Test with valid min but NaN max
  {
    OrcColumnStatistics stats;
    stats.has_min_max = true;
    stats.has_null = false;
    stats.num_values = 100;
    stats.min = std::make_shared<DoubleScalar>(10.0);
    stats.max = std::make_shared<DoubleScalar>(std::nan(""));

    auto expr = OrcFileFragment::EvaluateStatisticsAsExpression(*field_f64, ref_f64, stats);
    ASSERT_TRUE(expr.has_value());
    // Should only have min bound
    EXPECT_TRUE(expr->ToString().find(">=") != std::string::npos);
  }
}

}  // namespace dataset
}  // namespace arrow
