# Using CUDA

In your terminal, run:
```
cat >> ~/.bashrc <<'EOF'

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
```
Then restart the terminal or run `source ~/.bashrc`