# Quick Start for Next Agent

## CRITICAL: Read IMPLEMENTATION-GUIDE.md First

Before starting any implementation work, read `IMPLEMENTATION-GUIDE.md` which defines:
- How to use Parquet as a reference (inspiration, not modification)
- The comparison framework for ensuring quality
- Required outputs for each implementation session
- Non-negotiable constraints

## Task 0: Extend ORC Adapter

**Goal:** Add column statistics access to `cpp/src/arrow/adapters/orc/adapter.h`

**What to add:**
```cpp
// In adapter.h:
struct ColumnStatistics {
  bool has_null;
  int64_t num_values;
  bool has_minimum;
  bool has_maximum;
  std::shared_ptr<Scalar> minimum;
  std::shared_ptr<Scalar> maximum;
  bool is_statistics_deprecated;
};

Result<ColumnStatistics> GetStripeColumnStatistics(
    int64_t stripe, int64_t column);
```

**Implementation steps:**
1. Study liborc API for statistics access
2. Add struct and method declaration to adapter.h
3. Implement in adapter.cc (access liborc Reader's statistics)
4. Convert ORC statistics to Arrow format
5. Write unit test verifying statistics for int32/int64 columns

**Verification:**
```bash
cmake --build . --target arrow_orc
ctest -R orc  # All ORC tests should pass
```

## After Task 0: Phase 1 (Tasks 1-5)

### Task 1: OrcSchemaManifest structures
- File: `file_orc.h`
- Add OrcSchemaManifest and OrcSchemaField classes
- Similar to Parquet's SchemaManifest

### Task 2: BuildOrcSchemaManifest
- File: `file_orc.cc`
- Walk Arrow schema + ORC type tree
- Extract column indices from type tree

### Task 3: GetOrcColumnIndex
- File: `file_orc.cc`
- Resolve FieldRef -> ORC column index
- Handle nested fields

### Task 4: OrcFileFragment
- Files: `file_orc.h`, `file_orc.cc`
- Extend FileFragment with ORC-specific fields
- Add: metadata, manifest, statistics_cache

### Task 5: StripeStatisticsCache
- File: `file_orc.cc`
- Cache structure with stripe_guarantees
- Thread-safe with mutex

## Key Files

- **Implementation Guide:** `IMPLEMENTATION-GUIDE.md` (READ FIRST)
- **Task list:** `task_list.json` (36 tasks)
- **Specification:** `orc-predicate-pushdown.allium`
- **Parquet Reference:** `cpp/src/arrow/dataset/file_parquet.cc`

## Build Commands

```bash
# Configure (if needed)
cmake -S . -B build -DARROW_ORC=ON -DARROW_DATASET=ON

# Build ORC adapter
cmake --build build --target arrow_orc

# Build dataset module
cmake --build build --target arrow_dataset

# Run tests
ctest --test-dir build -R orc
```

## Getting Unstuck

1. **Can't access ORC statistics?**
   - Check liborc documentation: `orc/Reader.hh`
   - Look at existing adapter.cc for patterns
   - Alternative: access liborc directly from file_orc.cc

2. **Don't understand expression simplification?**
   - Study `file_parquet.cc` TestRowGroups function
   - Read Arrow compute expression docs
   - Start with simple case: literal true/false

3. **Thread safety confusion?**
   - Follow Parquet pattern: physical_schema_mutex_
   - Protect all cache reads/writes
   - Make operations idempotent

4. **Tests failing?**
   - Start with simplest test (single field, int32, >)
   - Hand-craft ORC file with known statistics
   - Verify stripe filtering manually

## Testing Strategy

1. **Unit tests** (per task)
   - Test each function in isolation
   - Mock/stub dependencies
   - Cover edge cases

2. **Integration tests** (after Task 20)
   - End-to-end: create ORC file -> filter -> verify results
   - Measure I/O reduction
   - Test with various predicates

3. **Performance benchmarks** (Task 33)
   - Compare to baseline (no filtering)
   - Measure cache benefit
   - Compare to Parquet performance

## Success = All 36 Tasks Complete

Check `task_list.json` regularly. Mark tasks "complete" only when fully verified.

---

**Start here:** Read `IMPLEMENTATION-GUIDE.md` -> Task 0 -> ORC adapter statistics APIs
