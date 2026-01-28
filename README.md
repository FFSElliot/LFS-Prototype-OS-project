# Log-Structured File System (LFS) Prototype

## Overview
This project is a Python simulation of a Log-Structured File System (LFS).
It demonstrates sequential log writing, inode-based metadata management,
checkpointing for crash recovery, and file system statistics reporting.

## Features
- Sequential (append-only) writes
- Inode-based file metadata
- File updates without overwriting old data
- Checkpoint creation for recovery
- Basic file system statistics
- Benchmark mode for performance testing

## How to Run

### Demo Mode
```bash
python lfs.py