# -*- coding: utf-8 -*-
"""Shared and domain-specific evidence, with an explicit lexical retrieval fallback.

Stage 1 selects up to 20 BM25 candidates from the shared library and requested
domains. Stage 2 re-scores only those candidates with BM25 and returns up to 5.
This is deterministic lexical retrieval, not dense retrieval or a cross-encoder.
Uploaded text is evidence data: no document can change a model/tool contract.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

from .config import PROJECT
from .knowledge import _sections, _terms

DOMAINS = ('shared', 'vrms', 'fatigue', 'emotion')
INDEX_VERSION = 'brain-hierarchical-bm25-1.0'
CHUNK_SIZE, OVERLAP = 600, 100
MAX_DOCUMENT_CHARACTERS = 200000


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _validate_text(value, field, maximum, allow_empty=False):
    if not isinstance(value, str) or len(value) > maximum or '\x00' in value:
        raise ValueError('Invalid ' + field)
    if not allow_empty and not value.strip():
        raise ValueError('Missing ' + field)
    return value.strip()


def _parse_document(text):
    """The bundled corpus uses a JSON metadata block between Markdown fences."""
    match = re.match(r'\A---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if match:
        meta = json.loads(match.group(1))
        if not isinstance(meta, dict):
            raise ValueError('Document metadata must be an object')
        return meta, text[match.end():]
    return {}, text


def _rank(chunks, terms):
    if not chunks or not terms:
        return []
    count = len(chunks)
    df = Counter()
    for chunk in chunks:
        df.update(set(chunk['_terms']))
    mean_length = sum(len(c['_terms']) for c in chunks) / float(count)
    ranked = []
    for chunk in chunks:
        tf = Counter(chunk['_terms'])
        score = 0.0
        for term in sorted(terms):
            frequency = tf.get(term, 0)
            if frequency:
                idf = math.log(1.0 + (count - df[term] + .5) / (df[term] + .5))
                norm = 1.2 * (.25 + .75 * len(chunk['_terms']) / max(mean_length, 1.0))
                score += idf * frequency * 2.2 / (frequency + norm)
        if score > 0 and math.isfinite(score):
            ranked.append((float(score), chunk))
    return sorted(ranked, key=lambda item: (-item[0], item[1]['id']))


class HierarchicalKnowledge:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.root = Path(self.cfg.get('brain_knowledge_root', Path(__file__).with_name('brain_knowledge')))
        output = Path(self.cfg.get('output_root', PROJECT / 'outputs'))
        self.user_root = Path(self.cfg.get('brain_knowledge_user_root', output / 'brain_knowledge'))
        self.lock = threading.RLock()
        self.chunks = []
        self.documents = []
        self.fingerprint = None

    def _ensure(self):
        files = []
        for library, root in (('bundled', self.root), ('uploaded', self.user_root)):
            for domain in DOMAINS:
                for path in sorted((root / domain).glob('*.md')):
                    if path.is_file():
                        text = path.read_text(encoding='utf-8')
                        if len(text) <= MAX_DOCUMENT_CHARACTERS + 5000:
                            files.append((library, domain, path.name, text))
        hashes = [(lib, domain, name, _sha(text)) for lib, domain, name, text in files]
        fingerprint = _sha(json.dumps(hashes, ensure_ascii=False))
        if fingerprint == self.fingerprint:
            return
        chunks, documents = [], []
        for library, domain, filename, raw in files:
            meta, body = _parse_document(raw)
            source = str(meta.get('source') or domain + '/' + filename)
            source_hash = str(meta.get('source_sha256') or _sha(raw))
            if not re.fullmatch(r'[a-fA-F0-9]{64}', source_hash):
                raise ValueError('Invalid knowledge source SHA256')
            document = {'domain': domain, 'layer': 'shared' if domain == 'shared' else 'domain',
                        'library': library, 'source': source, 'source_sha256': source_hash.lower(),
                        'document_sha256': _sha(raw), 'title': str(meta.get('title') or filename),
                        'doi': str(meta.get('doi') or ''), 'url': str(meta.get('url') or ''),
                        'page': str(meta.get('page') or ''),
                        'evidence_type': str(meta.get('evidence_type') or 'user_supplied'),
                        'version': str(meta.get('version') or '')}
            documents.append(document)
            for section, text in _sections(body):
                start = 0
                while start < len(text):
                    end = min(start + CHUNK_SIZE, len(text))
                    content = text[start:end].strip()
                    if content:
                        identity = '\n'.join((domain, source, source_hash, section, content))
                        citation = 'brain-kb:' + _sha(identity)[:20]
                        chunks.append({**document, 'section': section, 'text': content,
                                       'id': citation, 'citation_id': citation,
                                       'content_sha256': _sha(content), '_terms': _terms(section + ' ' + content)})
                    if end >= len(text):
                        break
                    start = end - OVERLAP
        self.chunks, self.documents, self.fingerprint = chunks, documents, fingerprint

    def summary(self):
        with self.lock:
            self._ensure()
            domains = {domain: {'sources': sum(d['domain'] == domain for d in self.documents),
                                'chunks': sum(c['domain'] == domain for c in self.chunks)}
                       for domain in DOMAINS}
            return {'status': 'ready' if self.chunks else 'insufficient', 'index_version': INDEX_VERSION,
                    'sources': len(self.documents), 'chunks': len(self.chunks), 'domains': domains,
                    'fingerprint': self.fingerprint, 'candidate_limit': 20, 'maximum_results': 5,
                    'retrieval_method': 'two_stage_bm25_lexical_fallback',
                    'dense_retrieval_available': False, 'cross_encoder_available': False,
                    'description': '共享库与选定领域库检索20个BM25候选，再用候选集BM25排序取最多5条；当前为词项检索。'}

    def search(self, query, domains=None, limit=5, method_only=False):
        query = _validate_text(query, 'query', 2000, allow_empty=True)
        domains = list(('vrms', 'fatigue', 'emotion') if domains is None else domains)
        if any(domain not in DOMAINS for domain in domains):
            raise ValueError('Unknown knowledge domain')
        domains = ['shared'] + sorted(set(domains) - {'shared'})
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError('Invalid retrieval limit')
        limit = max(0, min(5, limit))
        with self.lock:
            self._ensure()
            # Uploaded records can contain questionnaire answers. Raw-only
            # inference retrieves the reviewed bundled methods corpus only.
            pool = [c for c in self.chunks if c['domain'] in domains and
                    (not method_only or (c['library'] == 'bundled' and c['evidence_type'] == 'local_protocol'))]
            terms = set(_terms(query))
            candidates = _rank(pool, terms)[:20]
            reranked = _rank([c for _, c in candidates], terms)
            first_scores = {c['id']: score for score, c in candidates}
            results = [{**{k: v for k, v in chunk.items() if k != '_terms'},
                        'score': score, 'retrieval_score': first_scores[chunk['id']]}
                       for score, chunk in reranked[:limit]]
            return {'query': query, 'domains': domains, 'results': results,
                    'status': 'ok' if results else 'insufficient',
                    'reason': None if results else 'no matching local evidence',
                    'index_version': INDEX_VERSION, 'fingerprint': self.fingerprint,
                    'retrieval_method': 'two_stage_bm25_lexical_fallback',
                    'stages': [{'stage': 'shared_and_domain_retrieval', 'method': 'BM25',
                                'searched_chunks': len(pool), 'candidate_limit': 20,
                                'candidates': len(candidates), 'domains': domains},
                               {'stage': 'candidate_rerank', 'method': 'BM25 on candidate set',
                                'maximum_results': 5, 'returned': len(results),
                                'dense_or_cross_encoder': False}]}

    def add_document(self, domain, title, text, source, doi='', page='', evidence_type='user_supplied'):
        if domain not in DOMAINS:
            raise ValueError('Unknown knowledge domain')
        title = _validate_text(title, 'title', 200)
        text = _validate_text(text, 'document text', MAX_DOCUMENT_CHARACTERS)
        source = _validate_text(source, 'source', 1000)
        doi = _validate_text(doi, 'doi', 200, allow_empty=True)
        page = _validate_text(str(page), 'page', 100, allow_empty=True)
        # User text cannot declare itself to be a verified protocol or paper excerpt.
        if evidence_type != 'user_supplied':
            raise ValueError('Uploaded documents must use user_supplied evidence type')
        metadata = {'title': title, 'source': source, 'source_sha256': _sha(text),
                    'doi': doi, 'url': source if source.startswith(('https://', 'http://')) else '',
                    'page': page, 'evidence_type': evidence_type, 'version': ''}
        payload = '---\n' + json.dumps(metadata, ensure_ascii=False) + '\n---\n# ' + title + '\n\n' + text + '\n'
        identity = _sha(domain + '\n' + source + '\n' + text)
        with self.lock:
            directory = self.user_root / domain
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ('user_' + identity[:24] + '.md')
            temporary = directory / ('.' + uuid.uuid4().hex + '.tmp')
            try:
                temporary.write_text(payload, encoding='utf-8')
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self._ensure()
        return {'status': 'indexed', 'domain': domain, 'title': title,
                'document_id': identity, 'source': source, 'source_sha256': metadata['source_sha256'],
                'evidence_type': evidence_type}
