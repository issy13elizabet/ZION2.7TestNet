ls -la zion_universal_pool.py
echo "Error fields count:"
grep -c '"error": None' zion_universal_pool.py || echo "No error fields found"
echo "First few lines:"
head -10 zion_universal_pool.py | tail -5