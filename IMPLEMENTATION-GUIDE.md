# ORC Predicate Pushdown Implementation Guide

This document defines **how** to implement ORC predicate pushdown, using Parquet as the reference implementation. It establishes constraints, comparison frameworks, reuse rules, and required outputs for quality assurance.

**Read this before starting any implementation work.**

---

## Table of Contents

1. [The Parquet Reference Relationship](#the-parquet-reference-relationship)
2. [Non-Negotiable Constraints](#non-negotiable-constraints)
3. [Comparison Framework](#comparison-framework)
4. [Reuse & Sharing Rules](#reuse--sharing-rules)
5. [Test & Validation Strategy](#test--validation-strategy)
6. [Required Session Outputs](#required-session-outputs)
7. [Initial Parity Analysis](#initial-parity-analysis)
8. [Footguns Checklist](#footguns-checklist)
9. [Key Parquet Code References](#key-parquet-code-references)

---

## The Parquet Reference Relationship

The Parquet predicate pushdown reference is:

### Inspirational
Treat it as a proven blueprint for strategy, architecture, concurrency patterns, and feature completeness.

### A Source of Reusable Ideas and Patterns
You may recommend copying approaches and structure.

### Sometimes a Source of Reusable Code
You may suggest reusing generic utilities or abstractions if they are not Parquet-specific and can be shared cleanly (no tight coupling, no semantic mismatch).

### Never to Be Modified
Do not propose edits to reference files unless explicitly instructed. If you believe a change in shared code is necessary, propose an ORC-local alternative first, and only then suggest a shared abstraction as an optional follow-up.

---

## Non-Negotiable Constraints

1. **Do not touch the reference implementation** (Parquet predicate pushdown) unless explicitly instructed.

2. **Preserve semantics**: ORC pushdown must match ORC's encoding/reader semantics and Arrow's scan/filter semantics.

3. **Avoid accidental coupling**: Don't introduce Parquet-only assumptions into ORC (statistics formats, encodings, row-group logic, etc.).

4. **Keep concurrency safe**: Any parallel evaluation/IO must be race-free, deterministic in behavior, and consistent with Arrow's patterns.

5. **Conservative filtering**: Never exclude stripes that might contain matching rows. When in doubt, include the stripe.

---

## Comparison Framework

When comparing ORC vs Parquet pushdown, **always** evaluate these five dimensions:

### 1. Feature Surface & Semantics

| Aspect | Parquet Status | ORC Target | Notes |
|--------|---------------|------------|-------|
| Comparison predicates (=, !=, <, <=, >, >=) | Full support | Must implement | Core feature |
| Logical operators (AND, OR, NOT) | Full support | Must implement | Compound predicates |
| IN predicate | Supported | Must implement | Range intersection |
| IS NULL / IS VALID | Supported | Must implement | Null handling |
| Type coverage: int32, int64 | Supported | Phase 1 | Initial types |
| Type coverage: float32, float64 | Supported | Phase 2 | Float edge cases |
| Type coverage: string, binary | Supported | Phase 2 | Truncation handling |
| Type coverage: timestamp, date | Supported | Phase 2 | Unit conversion |
| Type coverage: decimal | Supported | Future | Complex |
| Nested types (struct/list/map) | Via SchemaManifest | Must implement | Column index mapping |
| Three-valued logic (NULL semantics) | Correct | Must match | UNKNOWN = include |

### 2. Pushdown Depth & Plan

| Aspect | Parquet | ORC Target |
|--------|---------|------------|
| Partition pruning (directory level) | Scanner handles | Same (no change) |
| Row group / stripe filtering | `FilterRowGroups()` | `FilterStripes()` |
| Sub-stripe (row index) | Not used | Not initially (future) |
| Expression binding | Defensive in `TestRowGroups` | Same pattern |
| Fallback on missing stats | Include row group | Include stripe |
| Fallback on corrupted stats | Include row group | Include stripe |

### 3. Statistics and Index Usage

| Aspect | Parquet | ORC | Difference |
|--------|---------|-----|------------|
| Statistics source | RowGroup column metadata | Stripe column statistics | API differs |
| Min/max availability | `has_min_max` flag | `has_minimum`, `has_maximum` | Similar |
| Null count | `null_count` field | `has_null`, `num_values` | ORC uses num_values=0 for all-null |
| Deprecated stats flag | Writer version check | `is_statistics_deprecated` | Similar concept |
| Bloom filters | Supported (separate) | Available in ORC | Future enhancement |
| Column index (page-level) | Supported | Row index (similar) | Future enhancement |

### 4. Concurrency & Performance Strategy

| Aspect | Parquet | ORC Target |
|--------|---------|------------|
| Cache protection | `physical_schema_mutex_` | Same pattern |
| Metadata caching | `metadata_`, `manifest_` | Same fields |
| Statistics caching | `statistics_expressions_[]` | `stripe_guarantees[]` |
| Column completion tracking | `statistics_expressions_complete_[]` | `statistics_complete[]` |
| Idempotent operations | Yes | Must maintain |
| Incremental cache population | Yes | Must implement |

### 5. Architecture & Extensibility

| Aspect | Parquet | ORC Target |
|--------|---------|------------|
| Fragment class | `ParquetFileFragment` | `OrcFileFragment` (NEW) |
| Schema manifest | `parquet::arrow::SchemaManifest` | `OrcSchemaManifest` (NEW) |
| Statistics to expression | `EvaluateStatisticsAsExpression()` | `DeriveFieldGuarantee()` |
| Row group testing | `TestRowGroups()` | `TestStripes()` |
| Row group filtering | `FilterRowGroups()` | `FilterStripes()` |
| Count optimization | `TryCountRows()` | `OrcTryCountRows()` |

---

## Reuse & Sharing Rules

When you see something strong in the Parquet reference, classify it into exactly one bucket:

### Idea Reuse (Preferred)
Replicate the design pattern or strategy in ORC-specific code.

**Examples:**
- Thread safety model with `physical_schema_mutex_`
- Incremental statistics cache population
- Defensive expression binding
- Conservative filtering invariants

### Infra Reuse
Reuse existing shared infrastructure if already designed to be format-agnostic.

**Examples:**
- `compute::SimplifyWithGuarantee()` - shared expression simplification
- `FileFormatFixtureMixin<T>` - test fixtures
- `compute::Expression` - expression representation
- `compute::Simplify()` - expression optimization

### Code Reuse (Only If Clean)
Suggest factoring or reusing code only if it is clearly generic and does not require changing the reference.

**For each reuse suggestion, explicitly state:**
1. Why it's reusable
2. What format-specific assumptions must be removed/avoided
3. Whether it requires new shared abstractions (and whether that would touch reference files)

---

## Test & Validation Strategy

### Reusable Test Infrastructure from Parquet

| Component | Location | Reusable? |
|-----------|----------|-----------|
| `FileFormatFixtureMixin<T>` | `test_util_internal.h` | YES - format-agnostic |
| `FileFormatScanMixin<T>` | `test_util_internal.h` | YES - format-agnostic |
| `OrcFormatHelper` | `file_orc_test.cc` | EXISTS - extend it |
| Expression builders | `compute/expression.h` | YES - shared |
| Test data generation | Format-specific | NO - ORC-specific needed |

### ORC-Specific Test Fixtures Needed

1. **Multi-stripe ORC file generator**
   - Create files with known statistics per stripe
   - Control min/max values, null counts
   - Support deprecated statistics flag

2. **Statistics edge case files**
   - All-null stripes (num_values = 0)
   - Single-value stripes (min = max)
   - Missing statistics
   - Corrupted statistics (min > max)

3. **Nested type test files**
   - Struct columns with leaf statistics
   - List columns
   - Map columns

### Test Parity Matrix

| Test Category | Parquet Has | ORC Needs |
|---------------|-------------|-----------|
| Basic scan tests | YES | YES (exists) |
| CountRows | YES | YES (exists) |
| CountRows with predicate pushdown | YES | **NO - ADD** |
| PredicatePushdown | YES | **NO - ADD** |
| PredicatePushdownRowGroupFragments | YES | **NO - ADD** |
| String column pushdown | YES | **FUTURE** |
| Duration column pushdown | YES | **FUTURE** |
| Multithreaded scan | YES | **NO - ADD** |
| Cached metadata | YES | **NO - ADD** |
| Explicit row group selection | YES | **NO - ADD** |

### Required New Tests for ORC

```cpp
// Tests to add to file_orc_test.cc

TEST_F(TestOrcFileFormat, CountRowsPredicatePushdown) { ... }
TEST_F(TestOrcFileFormat, CachedMetadata) { ... }
TEST_F(TestOrcFileFormat, MultithreadedScan) { ... }

TEST_P(TestOrcFileFormatScan, PredicatePushdown) { ... }
TEST_P(TestOrcFileFormatScan, PredicatePushdownStripeFragments) { ... }
TEST_P(TestOrcFileFormatScan, ExplicitStripeSelection) { ... }
```

---

## Required Session Outputs

Every implementation session **MUST** produce these sections:

### 1. Reference Snapshot
What parts of Parquet pushdown are most relevant to the current work.

### 2. ORC Current State
What exists, what changed recently, and what's under review.

### 3. Parity & Gaps Table
| Feature | Parquet | ORC | Status |
|---------|---------|-----|--------|
| ... | ... | ... | Parity/Missing/Different-by-design |

### 4. Reuse Plan
Ideas/infra/code reuse suggestions with constraints.

### 5. Risk Register
- Correctness risks
- Performance risks
- Concurrency risks

### 6. Action Checklist
Prioritized steps:
- P0: Correctness
- P1: Tests
- P2: Performance
- P3: Cleanup

### 7. Test Matrix
Predicate types × data types × metadata availability × edge cases.

---

## Initial Parity Analysis

### Current State Comparison

| Metric | Parquet | ORC | Gap |
|--------|---------|-----|-----|
| Header file lines | 410 | 75 | 5.5x |
| Implementation lines | 1200 | 233 | 5.1x |
| Test file lines | 999 | 96 | 10.4x |
| Fragment class | `ParquetFileFragment` (78 lines) | **MISSING** | Must create |
| Schema manifest | `parquet::arrow::SchemaManifest` | **MISSING** | Must create |
| Predicate pushdown tests | 8+ tests | 0 | Must add |

### Parquet Components to Mirror in ORC

| Parquet Component | Lines | ORC Equivalent | Priority |
|-------------------|-------|----------------|----------|
| `ParquetFileFragment` class | ~78 | `OrcFileFragment` | P0 |
| `TestRowGroups()` | ~50 | `TestStripes()` | P0 |
| `FilterRowGroups()` | ~15 | `FilterStripes()` | P0 |
| `TryCountRows()` | ~30 | `OrcTryCountRows()` | P1 |
| `EvaluateStatisticsAsExpression()` | ~80 | `DeriveFieldGuarantee()` | P0 |
| `EnsureCompleteMetadata()` | ~70 | `EnsureFileMetadataCached()` | P0 |
| Statistics caching members | ~10 | Same pattern | P0 |
| Thread safety (mutex) | Throughout | Same pattern | P0 |

### Key Semantic Differences

| Aspect | Parquet | ORC | Implementation Impact |
|--------|---------|-----|----------------------|
| Unit of filtering | Row Group | Stripe | Terminology only |
| Column indexing | Schema-ordered | Depth-first pre-order (col 0 = root) | Must handle offset |
| Null detection | `null_count = num_values` | `num_values = 0` | Different check |
| Statistics struct | `parquet::Statistics` | liborc statistics types | Different API |
| Manifest source | `parquet::arrow::SchemaManifest` | ORC type tree | Must build custom |

---

## Footguns Checklist

These edge cases can cause correctness bugs. Address each explicitly:

### Numeric Types
- [ ] **NaN handling** (float/double): NaN in statistics makes min/max unusable
- [ ] **Signed zero**: -0.0 == +0.0 but may appear differently in stats
- [ ] **Infinity**: +Inf/-Inf are valid min/max values
- [ ] **Overflow**: Statistics computation may overflow for large values
- [ ] **Decimal precision**: Scale/precision must match

### String/Binary Types
- [ ] **Truncation**: ORC may truncate long strings in statistics
- [ ] **Collation**: String ordering depends on encoding
- [ ] **Empty strings**: "" vs null distinction

### Temporal Types
- [ ] **Timestamp units**: Seconds vs milliseconds vs microseconds vs nanoseconds
- [ ] **Timezone handling**: UTC vs local time
- [ ] **Date boundaries**: Handling of dates before epoch

### Null Handling
- [ ] **Three-valued logic**: UNKNOWN != FALSE
- [ ] **All-null columns**: num_values = 0 detection
- [ ] **Null in predicates**: `x = NULL` is UNKNOWN, not FALSE

### Statistics Reliability
- [ ] **Deprecated statistics**: Old ORC writers had bugs
- [ ] **Missing statistics**: Not all columns have stats
- [ ] **Corrupted statistics**: min > max should be rejected
- [ ] **Empty stripes**: num_rows = 0 edge case

### Concurrency
- [ ] **Race conditions**: Multiple threads updating cache
- [ ] **Deadlocks**: Lock ordering
- [ ] **Idempotency**: Repeated operations must be safe

---

## Key Parquet Code References

Study these specific locations in the Parquet implementation:

### ParquetFileFragment Class
**File:** `cpp/src/arrow/dataset/file_parquet.h:158-235`

Key members to mirror:
```cpp
std::optional<std::vector<int>> row_groups_;           // -> stripes_
std::vector<compute::Expression> statistics_expressions_;  // -> stripe_guarantees_
std::vector<bool> statistics_expressions_complete_;    // -> statistics_complete_
std::shared_ptr<parquet::FileMetaData> metadata_;      // -> OrcFileMetadata
std::shared_ptr<parquet::arrow::SchemaManifest> manifest_;  // -> OrcSchemaManifest
```

### TestRowGroups Implementation
**File:** `cpp/src/arrow/dataset/file_parquet.cc:933-983`

Pattern to follow:
1. Lock mutex
2. Simplify predicate with partition expression
3. Check satisfiability (early exit)
4. Resolve predicate fields
5. For uncached columns: load statistics, derive guarantees
6. For each row group: simplify predicate with guarantee
7. Return per-row-group expressions

### FilterRowGroups Implementation
**File:** `cpp/src/arrow/dataset/file_parquet.cc:918-931`

Simple wrapper:
1. Call `TestRowGroups()`
2. Select row groups where expression is satisfiable

### TryCountRows Implementation
**File:** `cpp/src/arrow/dataset/file_parquet.cc:986-1010`

Optimization:
1. If no field refs: count = num_rows or 0
2. Call `TestRowGroups()`
3. Sum row counts for `literal(true)` groups
4. Return null if any group is not literal(true/false)

### Thread Safety Pattern
**File:** `cpp/src/arrow/dataset/file_parquet.cc`

Locations using `physical_schema_mutex_`:
- Line 798: `metadata()` accessor
- Line 803: `EnsureCompleteMetadata()`
- Line 923: `FilterRowGroups()`
- Line 935: `TestRowGroups()`

### Test Patterns
**File:** `cpp/src/arrow/dataset/file_parquet_test.cc`

Key tests to mirror:
- `CountRowsPredicatePushdown` (line 307)
- `PredicatePushdown` (line 639)
- `PredicatePushdownRowGroupFragments` (line 694)
- `CachedMetadata` (line 378)
- `MultithreadedScan` (line 436)

---

## Operating Mode

When implementing ORC predicate pushdown:

1. **Default to analyzing** the ORC-related code and comparing against Parquet patterns
2. **Produce structured comparisons** using the framework above
3. **Work autonomously**: identify gaps, propose solutions, validate correctness
4. **Never wait** for explicit direction on what to compare
5. **Always end with actionable steps**

Your goal is to ensure ORC predicate pushdown achieves a **high-quality, idiomatic implementation** that matches or intentionally diverges from the Parquet reference with clear justification.

---

## Quick Reference: File Locations

| Purpose | Parquet | ORC |
|---------|---------|-----|
| Header | `cpp/src/arrow/dataset/file_parquet.h` | `cpp/src/arrow/dataset/file_orc.h` |
| Implementation | `cpp/src/arrow/dataset/file_parquet.cc` | `cpp/src/arrow/dataset/file_orc.cc` |
| Tests | `cpp/src/arrow/dataset/file_parquet_test.cc` | `cpp/src/arrow/dataset/file_orc_test.cc` |
| ORC Adapter | - | `cpp/src/arrow/adapters/orc/adapter.h` |
| Specification | - | `orc-predicate-pushdown.allium` |
