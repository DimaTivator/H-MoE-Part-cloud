from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from src.data.fineweb_packed import PackedFineWebTrainReader
from src.data.fineweb_replay import (
    FineWebReplayTrainReader,
    FineWebSerialReplayTrainReader,
    blocks_sha256,
)


class FineWebReplayTrainReaderTest(unittest.TestCase):
    def _build_reader(self, root: Path, rank_blocks: list[np.ndarray]):
        ranks = []
        for rank, blocks in enumerate(rank_blocks):
            path = root / f"rank{rank}.uint16"
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            path.write_bytes(payload)
            ranks.append(
                {
                    "rank": rank,
                    "file": path.name,
                    "blocks": len(blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        metadata = {
            "world_size": 2,
            "batch_size": 2,
            "sequence_length": 2,
            "ranks": ranks,
        }
        sources = [
            PackedFineWebTrainReader(
                root,
                metadata,
                rank=rank,
                world_size=2,
                batch_size=2,
                sequence_length=2,
            )
            for rank in range(2)
        ]
        return FineWebReplayTrainReader(
            sources,
            batch_size=4,
            sequence_length=2,
        )

    def test_concatenates_source_rank_batches_in_rank_order(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            reader = self._build_reader(Path(temporary), rank_blocks)
            x, y = reader.sample_batch()

        expected_blocks = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [101, 102, 103], [104, 105, 106]]
        )
        torch.testing.assert_close(x, expected_blocks[:, :-1])
        torch.testing.assert_close(y, expected_blocks[:, 1:])

    def test_logs_stable_hashes_and_restores_both_sources(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(os.environ, {"FINEWEB_BATCH_HASH_STEPS": "1"}):
                reader = self._build_reader(root, rank_blocks)
                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    reader.sample_batch()
                state = reader.state_dict()

                restored = self._build_reader(root, rank_blocks)
                restored.load_state_dict(state)
                restored_x, restored_y = restored.sample_batch()

        first_blocks = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [101, 102, 103], [104, 105, 106]]
        )
        self.assertIn(f"combined={blocks_sha256(first_blocks)}", output.getvalue())
        expected_next = torch.tensor(
            [[7, 8, 9], [10, 11, 12], [107, 108, 109], [110, 111, 112]]
        )
        torch.testing.assert_close(restored_x, expected_next[:, :-1])
        torch.testing.assert_close(restored_y, expected_next[:, 1:])


class FineWebSerialReplayTrainReaderTest(FineWebReplayTrainReaderTest):
    def _build_reader(self, root: Path, rank_blocks: list[np.ndarray]):
        ranks = []
        for rank, blocks in enumerate(rank_blocks):
            path = root / f"rank{rank}.uint16"
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            path.write_bytes(payload)
            ranks.append(
                {
                    "rank": rank,
                    "file": path.name,
                    "blocks": len(blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        metadata = {
            "world_size": 2,
            "batch_size": 2,
            "sequence_length": 2,
            "ranks": ranks,
        }
        sources = [
            PackedFineWebTrainReader(
                root,
                metadata,
                rank=rank,
                world_size=2,
                batch_size=2,
                sequence_length=2,
            )
            for rank in range(2)
        ]
        return FineWebSerialReplayTrainReader(
            sources,
            batch_size=2,
            sequence_length=2,
        )

    def test_concatenates_source_rank_batches_in_rank_order(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            reader = self._build_reader(Path(temporary), rank_blocks)
            rank0_x, rank0_y = reader.sample_batch()
            rank1_x, rank1_y = reader.sample_batch()

        expected_rank0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_rank1 = torch.tensor([[101, 102, 103], [104, 105, 106]])
        torch.testing.assert_close(rank0_x, expected_rank0[:, :-1])
        torch.testing.assert_close(rank0_y, expected_rank0[:, 1:])
        torch.testing.assert_close(rank1_x, expected_rank1[:, :-1])
        torch.testing.assert_close(rank1_y, expected_rank1[:, 1:])

    def test_logs_stable_hashes_and_restores_both_sources(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = self._build_reader(root, rank_blocks)
            reader.sample_batch()
            reader.sample_batch()
            reader.sample_batch()
            state = reader.state_dict()

            restored = self._build_reader(root, rank_blocks)
            restored.load_state_dict(state)
            restored_x, restored_y = restored.sample_batch()

        expected = torch.tensor([[107, 108, 109], [110, 111, 112]])
        torch.testing.assert_close(restored_x, expected[:, :-1])
        torch.testing.assert_close(restored_y, expected[:, 1:])


if __name__ == "__main__":
    unittest.main()
