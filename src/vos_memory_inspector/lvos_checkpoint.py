"""Immutable checkpoint body + recoverable journal + verified latest pointer.

ExclusiveWriter가 잡힌 run에서만 호출한다. 미확인 파일은 삭제/덮어쓰지 않는다.
"""
from pathlib import Path
import hashlib
import io
import os
import uuid
import torch
from .collection_contract import require
from .training_storage import atomic_write, write_json, sha256, content_hash
from .lvos_contract import json_read, load_complete


class CheckpointStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = identity

    def boundary(self, stage):
        """Fault injection seam; production에서는 추가 작업 없음."""

    def _validate(self, path, entry, epoch):
        require(path.is_file() and path.stat().st_size == entry['bytes'] and
                sha256(path) == entry['sha256'], 'CHECKPOINT_TRANSACTION_CHECKSUM', str(path))
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(value['identity'] == self.identity and value['epoch'] == epoch and
                value['boundary'] == 'complete_epoch', 'CHECKPOINT_TRANSACTION_IDENTITY')
        return value

    def _finish(self, journal):
        require(journal['identity_digest'] == content_hash(self.identity), 'FOREIGN_CHECKPOINT_JOURNAL')
        body = self.root / journal['entry']['path']
        staged = self.root / journal['staged_path']
        require(body.resolve().is_relative_to(self.root.resolve()) and
                staged.resolve().is_relative_to(self.root.resolve()), 'CHECKPOINT_PATH')
        if not body.exists() and not staged.exists():
            return None  # before body write: journal remains immutable evidence.
        if not body.exists():
            self._validate(staged, journal['entry'], journal['epoch'])
            os.replace(staged, body)
            if os.name != 'nt':
                fd = os.open(body.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        self._validate(body, journal['entry'], journal['epoch'])
        self.boundary('after_body')
        marker = Path(str(body) + '.complete.json')
        expected = {**journal['entry'], 'path':body.name, 'schema_version':'cmmt.ready_artifact.v1'}
        if marker.exists():
            require(json_read(marker) == expected, 'CHECKPOINT_MARKER_CONFLICT')
        else:
            write_json(marker, expected)
        self.boundary('after_marker')
        return journal['entry']

    def reconcile(self):
        recovery = []
        for path in sorted((self.root/'checkpoints/transactions').glob('*.json')):
            value = json_read(path)
            entry = self._finish(value)
            recovery.append({'journal':str(path.relative_to(self.root)),
                             'status':'body_missing_preserved' if entry is None else 'verified'})
        valid = []
        for path in sorted((self.root/'checkpoints').glob('epoch-*.ckpt')):
            value, entry = load_complete(path)  # corrupt/foreign/orphan evidence stops here.
            require(value['identity'] == self.identity and value['boundary'] == 'complete_epoch',
                    'FOREIGN_CHECKPOINT')
            require(path.name == f"epoch-{value['epoch']:05d}.ckpt", 'CHECKPOINT_EPOCH_PATH')
            pointer_entry = {k:entry[k] for k in ('path','sha256','bytes')}
            pointer_entry['path'] = 'checkpoints/'+entry['path']
            valid.append((value['epoch'], value, pointer_entry))
        if not valid:
            return None
        require([v[0] for v in valid] == list(range(valid[-1][0]+1)), 'CHECKPOINT_CHAIN_GAP')
        epoch, payload, entry = valid[-1]
        pointer = self.root/'last.ckpt.json'
        if pointer.exists():
            old = json_read(pointer)
            require(any(v[2] == old for v in valid), 'FOREIGN_OR_CORRUPT_LAST_POINTER')
        write_json(pointer, entry)
        write_json(self.root/'checkpoint_recovery.json', {'latest_epoch':epoch,'journals':recovery})
        self.boundary('after_pointer')
        return payload, entry

    def publish(self, payload):
        epoch = payload['epoch']
        target = self.root/f'checkpoints/epoch-{epoch:05d}.ckpt'
        require(not target.exists(), 'ARTIFACT_EXISTS', str(target))
        stream = io.BytesIO()
        torch.save(payload, stream)
        data = stream.getvalue()
        token = uuid.uuid4().hex
        entry = {'path':str(target.relative_to(self.root)).replace('\\','/'),
                 'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        journal = {'schema_version':'cmmt.checkpoint_transaction.v1','epoch':epoch,
                   'identity_digest':content_hash(self.identity),'entry':entry,
                   'staged_path':f'checkpoints/transactions/{token}.body'}
        write_json(self.root/f'checkpoints/transactions/{token}.json', journal)
        self.boundary('before_body')
        atomic_write(self.root/journal['staged_path'], lambda f:f.write(data))
        self._finish(journal)
        write_json(self.root/'last.ckpt.json',entry)
        self.boundary('after_pointer')
        return entry
