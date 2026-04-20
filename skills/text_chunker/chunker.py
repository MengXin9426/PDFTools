"""
文本分块器
基于断行分析和文档结构进行智能分块
"""

from typing import List, Dict
import re


class TextChunker:
    """文本分块器"""

    def __init__(self):
        """初始化分块器"""
        self.max_chunk_size = 1000  # 最大字符数
        self.overlap = 100  # 重叠字符数

    def chunk(self, extracted_text: Dict) -> List[Dict]:
        """
        对提取的文本进行分块

        Args:
            extracted_text: 提取的文本数据

        Returns:
            分块后的文本列表
        """
        print("\n文本分块...")

        chunks = []

        for page_data in extracted_text['pages']:
            page_chunks = self._chunk_page(page_data)
            chunks.extend(page_chunks)

        print(f"  ✓ 分为 {len(chunks)} 个文本块")
        return chunks

    def _chunk_page(self, page_data: Dict) -> List[Dict]:
        """
        对单页进行分块

        Args:
            page_data: 页面数据

        Returns:
            该页的分块列表
        """
        chunks = []
        current_chunk = {
            'page': page_data['page'],
            'content': [],
            'type': 'text'
        }
        # 与 ImageExtractor 命名 page{N}_img{M}.jpg 对齐（同页按块出现顺序）
        figure_slot = 0

        for block in page_data.get('blocks', []):
            if block['type'] == 'image':
                # 图片单独作为一个块
                if current_chunk['content']:
                    chunks.append(current_chunk.copy())
                    current_chunk['content'] = []

                figure_slot += 1
                chunks.append({
                    'page': page_data['page'],
                    'type': 'image',
                    'bbox': block['bbox'],
                    'figure_slot': figure_slot,
                })

            elif block['type'] == 'text':
                # 处理文本块
                lines = block.get('lines', [])

                for line in lines:
                    # 检测标题
                    if self._is_header(line):
                        if current_chunk['content']:
                            chunks.append(current_chunk.copy())
                            current_chunk['content'] = []

                        chunks.append({
                            'page': page_data['page'],
                            'type': 'header',
                            'content': [line]
                        })
                    else:
                        current_chunk['content'].append(line)

                        # 检查是否达到最大大小
                        if len('\n'.join(current_chunk['content'])) > self.max_chunk_size:
                            chunks.append(current_chunk.copy())
                            current_chunk['content'] = []

        # 添加最后一个块
        if current_chunk['content']:
            chunks.append(current_chunk)

        return chunks

    def _is_header(self, line: str) -> bool:
        """
        判断是否为标题行

        Args:
            line: 文本行

        Returns:
            是否为标题
        """
        # 简单规则：全大写、数字开头、短行等
        line = line.strip()

        if not line:
            return False

        # 全大写且长度适中
        if line.isupper() and 3 < len(line) < 80:
            return True

        # 数字开头的行（如 "1. Introduction"）
        if re.match(r'^\d+\.\s', line):
            return True

        # 常见标题词
        header_keywords = [
            'abstract', 'introduction', 'conclusion',
            'references', 'method', 'results', 'discussion'
        ]
        if line.lower() in header_keywords:
            return True

        return False

    def merge_for_translation(self, chunks: List[Dict]) -> List[Dict]:
        """
        合并小块以便翻译

        Args:
            chunks: 分块列表

        Returns:
            合并后的分块列表
        """
        merged = []
        current_group = []

        for chunk in chunks:
            if chunk['type'] == 'image':
                if current_group:
                    merged.append({
                        'type': 'text_group',
                        'chunks': current_group
                    })
                    current_group = []

                merged.append(chunk)

            elif chunk['type'] == 'header':
                if current_group:
                    merged.append({
                        'type': 'text_group',
                        'chunks': current_group
                    })
                    current_group = []

                merged.append(chunk)

            else:  # text
                current_group.append(chunk)

                # 检查大小（content 为行列表，需按字符数累计）
                def _text_chunk_chars(c: Dict) -> int:
                    body = c.get('content') or []
                    if isinstance(body, str):
                        return len(body)
                    if isinstance(body, list):
                        return sum(len(line) for line in body if isinstance(line, str))
                    return 0

                total_chars = sum(_text_chunk_chars(c) for c in current_group)
                if total_chars > self.max_chunk_size:
                    merged.append({
                        'type': 'text_group',
                        'chunks': current_group
                    })
                    current_group = []

        if current_group:
            merged.append({
                'type': 'text_group',
                'chunks': current_group
            })

        return merged
