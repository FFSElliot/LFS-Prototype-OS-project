"""
Log-Structured File System (LFS) Implementation
================================================

A complete implementation of a log-structured file system that demonstrates:
- Sequential write optimization
- Segment-based garbage collection
- Inode mapping for fast lookups
- Crash recovery with checkpointing
- Performance comparison with traditional file systems
"""

import os
import json
import time
import struct
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

BLOCK_SIZE = 4096          # 4KB blocks
SEGMENT_SIZE = 1024 * 1024 # 1MB segments
BLOCKS_PER_SEGMENT = SEGMENT_SIZE // BLOCK_SIZE
INODE_SIZE = 256           # Size of inode structure
CHECKPOINT_INTERVAL = 10   # Checkpoint every N writes
GC_THRESHOLD = 0.7         # Trigger GC when segment is 70% full

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Inode:
    """Represents file metadata"""
    inode_num: int
    file_size: int
    block_count: int
    block_addresses: List[Tuple[int, int]]  # List of (segment, offset)
    created_time: float
    modified_time: float
    is_directory: bool = False
    
    def to_bytes(self) -> bytes:
        """Serialize inode to bytes"""
        data = {
            'inode_num': self.inode_num,
            'file_size': self.file_size,
            'block_count': self.block_count,
            'block_addresses': self.block_addresses,
            'created_time': self.created_time,
            'modified_time': self.modified_time,
            'is_directory': self.is_directory
        }
        return pickle.dumps(data)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'Inode':
        """Deserialize inode from bytes"""
        d = pickle.loads(data)
        return Inode(**d)


@dataclass
class SegmentSummary:
    """Summary information for a segment"""
    segment_num: int
    live_blocks: int
    total_blocks: int
    utilization: float
    
    def update_utilization(self):
        """Recalculate utilization percentage"""
        if self.total_blocks > 0:
            self.utilization = self.live_blocks / self.total_blocks
        else:
            self.utilization = 0.0


@dataclass
class LogEntry:
    """Entry in the sequential log"""
    entry_type: str  # 'inode', 'data', 'checkpoint'
    inode_num: int
    data: bytes
    timestamp: float
    checksum: str


# ============================================================================
# DISK ABSTRACTION LAYER
# ============================================================================

class VirtualDisk:
    """Simulates a disk using a file"""
    
    def __init__(self, filename: str, size_mb: int = 100):
        self.filename = filename
        self.size = size_mb * 1024 * 1024
        self.num_segments = size_mb  # 1MB per segment
        
        # Create or open disk file
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.write(b'\x00' * self.size)
        
        self.file = open(filename, 'r+b')
        
    def read_block(self, segment: int, offset: int, size: int = BLOCK_SIZE) -> bytes:
        """Read a block from disk"""
        position = segment * SEGMENT_SIZE + offset
        self.file.seek(position)
        return self.file.read(size)
    
    def write_block(self, segment: int, offset: int, data: bytes):
        """Write a block to disk"""
        position = segment * SEGMENT_SIZE + offset
        self.file.seek(position)
        self.file.write(data)
        self.file.flush()
    
    def close(self):
        """Close the disk file"""
        self.file.close()


# ============================================================================
# INODE MAP
# ============================================================================

class InodeMap:
    """Maps inode numbers to their current disk locations"""
    
    def __init__(self):
        # inode_num -> (segment, offset)
        self.map: Dict[int, Tuple[int, int]] = {}
        self.next_inode_num = 1
    
    def allocate_inode(self) -> int:
        """Allocate a new inode number"""
        inode_num = self.next_inode_num
        self.next_inode_num += 1
        return inode_num
    
    def update(self, inode_num: int, segment: int, offset: int):
        """Update inode location"""
        self.map[inode_num] = (segment, offset)
    
    def lookup(self, inode_num: int) -> Optional[Tuple[int, int]]:
        """Look up inode location"""
        return self.map.get(inode_num)
    
    def remove(self, inode_num: int):
        """Remove inode from map"""
        if inode_num in self.map:
            del self.map[inode_num]
    
    def to_bytes(self) -> bytes:
        """Serialize inode map"""
        return pickle.dumps(self.map)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'InodeMap':
        """Deserialize inode map"""
        imap = InodeMap()
        imap.map = pickle.loads(data)
        if imap.map:
            imap.next_inode_num = max(imap.map.keys()) + 1
        return imap


# ============================================================================
# LOG-STRUCTURED FILE SYSTEM
# ============================================================================

class LogStructuredFileSystem:
    """Main LFS implementation"""
    
    def __init__(self, disk_file: str = "lfs_disk.img", disk_size_mb: int = 100):
        self.disk = VirtualDisk(disk_file, disk_size_mb)
        self.inode_map = InodeMap()
        
        # Current write position
        self.current_segment = 0
        self.current_offset = 0
        
        # Segment management
        self.segment_summaries = [
            SegmentSummary(i, 0, 0, 0.0) 
            for i in range(self.disk.num_segments)
        ]
        
        # Track which blocks are live (not obsolete)
        self.live_blocks: Dict[Tuple[int, int], int] = {}  # (seg, off) -> inode_num
        
        # Statistics
        self.stats = {
            'total_writes': 0,
            'sequential_writes': 0,
            'gc_runs': 0,
            'checkpoints': 0
        }
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    # ========================================================================
    # WRITE PATH
    # ========================================================================
    
    def create_file(self, filename: str, data: bytes) -> int:
        """Create a new file and write data"""
        inode_num = self.inode_map.allocate_inode()
        return self.write_file(inode_num, data)
    
    def write_file(self, inode_num: int, data: bytes) -> int:
        """Write data to a file (append to log)"""
        # Mark old blocks as obsolete
        old_location = self.inode_map.lookup(inode_num)
        if old_location:
            old_inode = self._read_inode(inode_num)
            if old_inode:
                for seg, off in old_inode.block_addresses:
                    self._mark_block_obsolete(seg, off)
        
        # Write data blocks sequentially
        block_addresses = []
        offset = 0
        
        while offset < len(data):
            block_data = data[offset:offset + BLOCK_SIZE]
            
            # Pad last block if necessary
            if len(block_data) < BLOCK_SIZE:
                block_data += b'\x00' * (BLOCK_SIZE - len(block_data))
            
            seg, off = self._append_to_log(block_data, 'data')
            block_addresses.append((seg, off))
            offset += BLOCK_SIZE
        
        # Create inode
        inode = Inode(
            inode_num=inode_num,
            file_size=len(data),
            block_count=len(block_addresses),
            block_addresses=block_addresses,
            created_time=time.time(),
            modified_time=time.time()
        )
        
        # Write inode to log
        inode_data = inode.to_bytes()
        seg, off = self._append_to_log(inode_data, 'inode')
        
        # Update inode map
        self.inode_map.update(inode_num, seg, off)
        
        # Mark all blocks as live
        for seg, off in block_addresses:
            self._mark_block_live(seg, off, inode_num)
        self._mark_block_live(seg, off, inode_num)  # inode itself
        
        self.stats['total_writes'] += 1
        self.stats['sequential_writes'] += 1
        
        # Trigger checkpoint periodically
        if self.stats['total_writes'] % CHECKPOINT_INTERVAL == 0:
            self.checkpoint()
        
        # Trigger GC if needed
        self._check_garbage_collection()
        
        return inode_num
    
    def _append_to_log(self, data: bytes, entry_type: str) -> Tuple[int, int]:
        """Append data to the sequential log"""
        # Check if current segment is full
        if self.current_offset + len(data) > SEGMENT_SIZE:
            self.current_segment = (self.current_segment + 1) % self.disk.num_segments
            self.current_offset = 0
        
        # Write to disk
        segment = self.current_segment
        offset = self.current_offset
        
        self.disk.write_block(segment, offset, data)
        
        # Update segment summary
        summary = self.segment_summaries[segment]
        summary.total_blocks += 1
        summary.update_utilization()
        
        # Advance write pointer
        self.current_offset += len(data)
        
        return segment, offset
    
    # ========================================================================
    # READ PATH
    # ========================================================================
    
    def read_file(self, inode_num: int) -> Optional[bytes]:
        """Read file data"""
        inode = self._read_inode(inode_num)
        if not inode:
            return None
        
        # Read all data blocks
        data = b''
        for seg, off in inode.block_addresses:
            block_data = self.disk.read_block(seg, off)
            data += block_data
        
        # Return only the actual file size
        return data[:inode.file_size]
    
    def _read_inode(self, inode_num: int) -> Optional[Inode]:
        """Read inode from disk"""
        location = self.inode_map.lookup(inode_num)
        if not location:
            return None
        
        seg, off = location
        inode_data = self.disk.read_block(seg, off, INODE_SIZE)
        
        try:
            return Inode.from_bytes(inode_data)
        except:
            return None
    
    def delete_file(self, inode_num: int):
        """Delete a file"""
        inode = self._read_inode(inode_num)
        if inode:
            # Mark all blocks as obsolete
            for seg, off in inode.block_addresses:
                self._mark_block_obsolete(seg, off)
            
            # Mark inode block as obsolete
            location = self.inode_map.lookup(inode_num)
            if location:
                self._mark_block_obsolete(*location)
            
            # Remove from inode map
            self.inode_map.remove(inode_num)
    
    # ========================================================================
    # GARBAGE COLLECTION
    # ========================================================================
    
    def _check_garbage_collection(self):
        """Check if garbage collection is needed"""
        for summary in self.segment_summaries:
            if summary.utilization < GC_THRESHOLD and summary.total_blocks > 0:
                self._clean_segment(summary.segment_num)
                break
    
    def _clean_segment(self, segment_num: int):
        """Clean a segment by copying live data"""
        print(f"CLEAN: Cleaning segment {segment_num}...")
        
        summary = self.segment_summaries[segment_num]
        
        # Find all live blocks in this segment
        live_data = []
        for (seg, off), inode_num in list(self.live_blocks.items()):
            if seg == segment_num:
                # Read the live block
                block_data = self.disk.read_block(seg, off)
                live_data.append((inode_num, block_data))
        
        # Copy live data to new location
        for inode_num, block_data in live_data:
            new_seg, new_off = self._append_to_log(block_data, 'data')
            
            # Update inode map if this is an inode block
            old_location = self.inode_map.lookup(inode_num)
            if old_location and old_location[0] == segment_num:
                self.inode_map.update(inode_num, new_seg, new_off)
            
            # Update live blocks tracking
            for (seg, off), inum in list(self.live_blocks.items()):
                if seg == segment_num and inum == inode_num:
                    del self.live_blocks[(seg, off)]
                    self.live_blocks[(new_seg, new_off)] = inode_num
        
        # Reset segment
        summary.live_blocks = 0
        summary.total_blocks = 0
        summary.update_utilization()
        
        self.stats['gc_runs'] += 1
        print(f"CHECKPOINT: Segment {segment_num} cleaned")
    
    def _mark_block_live(self, segment: int, offset: int, inode_num: int):
        """Mark a block as live (contains current data)"""
        self.live_blocks[(segment, offset)] = inode_num
        self.segment_summaries[segment].live_blocks += 1
        self.segment_summaries[segment].update_utilization()
    
    def _mark_block_obsolete(self, segment: int, offset: int):
        """Mark a block as obsolete (old version)"""
        if (segment, offset) in self.live_blocks:
            del self.live_blocks[(segment, offset)]
            self.segment_summaries[segment].live_blocks -= 1
            self.segment_summaries[segment].update_utilization()
    
    # ========================================================================
    # CRASH RECOVERY
    # ========================================================================
    
    def checkpoint(self):
        """Create a checkpoint for crash recovery"""
        print("CHECKPOINT: Creating checkpoint...")
        
        checkpoint_data = {
            'inode_map': self.inode_map.to_bytes(),
            'current_segment': self.current_segment,
            'current_offset': self.current_offset,
            'segment_summaries': pickle.dumps(self.segment_summaries),
            'live_blocks': pickle.dumps(self.live_blocks),
            'stats': self.stats,
            'timestamp': time.time()
        }
        
        # Write checkpoint to a reserved area (segment 0, special region)
        checkpoint_bytes = pickle.dumps(checkpoint_data)
        
        # In a real system, this would go to a special checkpoint region
        with open('lfs_checkpoint.dat', 'wb') as f:
            f.write(checkpoint_bytes)
        
        self.stats['checkpoints'] += 1
        print("Checkpoint created")
    
    def _load_checkpoint(self):
        """Load the most recent checkpoint"""
        try:
            with open('lfs_checkpoint.dat', 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.inode_map = InodeMap.from_bytes(checkpoint_data['inode_map'])
            self.current_segment = checkpoint_data['current_segment']
            self.current_offset = checkpoint_data['current_offset']
            self.segment_summaries = pickle.loads(checkpoint_data['segment_summaries'])
            self.live_blocks = pickle.loads(checkpoint_data['live_blocks'])
            self.stats = checkpoint_data['stats']
            
            print(" Loaded checkpoint from", 
                time.ctime(checkpoint_data['timestamp']))
        except FileNotFoundError:
            print("INFO: No checkpoint found, starting fresh")
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get file system statistics"""
        total_space = self.disk.num_segments * SEGMENT_SIZE
        used_space = sum(s.total_blocks * BLOCK_SIZE for s in self.segment_summaries)
        
        return {
            **self.stats,
            'total_space_mb': total_space / (1024 * 1024),
            'used_space_mb': used_space / (1024 * 1024),
            'utilization': (used_space / total_space) * 100 if total_space > 0 else 0,
            'num_files': len(self.inode_map.map),
            'segments': len(self.segment_summaries)
        }
    
    def close(self):
        """Clean shutdown"""
        self.checkpoint()
        self.disk.close()


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_lfs():
    """Benchmark LFS performance"""
    print("\n" + "="*60)
    print("LOG-STRUCTURED FILE SYSTEM BENCHMARK")
    print("="*60 + "\n")
    
    # Initialize file system
    lfs = LogStructuredFileSystem("benchmark_lfs.img", disk_size_mb=50)
    
    # Test 1: Sequential writes
    print("Test 1: Sequential Writes")
    print("-" * 40)
    
    num_files = 20
    file_size = 8192  # 8KB per file
    
    start_time = time.time()
    inode_nums = []
    
    for i in range(num_files):
        data = f"File {i} content ".encode() * (file_size // 20)
        data = data[:file_size]
        inode = lfs.create_file(f"file_{i}.txt", data)
        inode_nums.append(inode)
    
    write_time = time.time() - start_time
    
    print(f"CHECKPOINT: Wrote {num_files} files ({file_size} bytes each)")
    print(f"  Total time: {write_time:.3f}s")
    print(f"  Average: {write_time/num_files*1000:.2f}ms per file")
    print(f"  Throughput: {(num_files * file_size / 1024 / 1024) / write_time:.2f} MB/s")
    
    # Test 2: Random reads
    print("\nTest 2: Random Reads")
    print("-" * 40)
    
    import random
    random.shuffle(inode_nums)
    
    start_time = time.time()
    
    for inode in inode_nums[:10]:
        data = lfs.read_file(inode)
    
    read_time = time.time() - start_time
    
    print(f"CHECKPOINT: Read 10 random files")
    print(f"  Total time: {read_time:.3f}s")
    print(f"  Average: {read_time/10*1000:.2f}ms per file")
    
    # Test 3: Updates (creates new versions)
    print("\nTest 3: File Updates")
    print("-" * 40)
    
    start_time = time.time()
    
    for inode in inode_nums[:5]:
        new_data = b"Updated content " * 512
        lfs.write_file(inode, new_data[:file_size])
    
    update_time = time.time() - start_time
    
    print(f"CHECKPOINT: Updated 5 files")
    print(f"  Total time: {update_time:.3f}s")
    print(f"  Average: {update_time/5*1000:.2f}ms per update")
    
    # Display statistics
    print("\nFile System Statistics")
    print("-" * 40)
    stats = lfs.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    lfs.close()
    print("\n Benchmark complete")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate LFS features"""
    print("\n" + "="*60)
    print("LOG-STRUCTURED FILE SYSTEM DEMO")
    print("="*60 + "\n")
    
    # Create file system
    print("1. Initializing file system...")
    lfs = LogStructuredFileSystem("demo_lfs.img", disk_size_mb=10)
    
    # Create files
    print("\n2. Creating files...")
    file1 = lfs.create_file("document.txt", b"Hello, World!" * 100)
    file2 = lfs.create_file("image.dat", b"\x89PNG" * 500)
    print(f"   Created file1 (inode {file1})")
    print(f"   Created file2 (inode {file2})")
    
    # Read file
    print("\n3. Reading file...")
    data = lfs.read_file(file1)
    print(f"   Read {len(data)} bytes from file1")
    
    # Update file (creates new version in log)
    print("\n4. Updating file...")
    lfs.write_file(file1, b"Updated content!" * 100)
    print(f"   Updated file1 (old version marked obsolete)")
    
    # Create more files to trigger GC
    print("\n5. Creating more files to trigger GC...")
    for i in range(10):
        lfs.create_file(f"test_{i}.txt", f"Test file {i}".encode() * 200)
    
    # Show stats
    print("\n6. File System Statistics:")
    print("-" * 40)
    stats = lfs.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Clean up
    lfs.close()
    print("\n Demo complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_lfs()
    else:
        demo()
        print("\nRun with 'python lfs.py benchmark' for performance tests")