-- schema_patch_v2.sql
-- Run this ONLY if you already ran schema.sql from Day 1
-- If you're starting fresh, add this column directly to schema.sql instead

ALTER TABLE users ADD COLUMN IF NOT EXISTS std_txn_amount NUMERIC(10,2) DEFAULT 0;

-- Verify
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'users' 
ORDER BY ordinal_position;
