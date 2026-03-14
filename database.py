"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================
包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
"""

import os
import re
from typing import Optional, List

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "")

# 搜索权重（向量搜索加入后可重新分配）
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def init_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts 
            ON memories 
            USING gin(to_tsvector('simple', content));
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations (session_id, created_at);
        """)

        # Bot 持久化：对话历史
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_conversation_history (
                id         SERIAL PRIMARY KEY,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Bot 持久化：DDL 任务
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_ddl_list (
                id         SERIAL PRIMARY KEY,
                title      TEXT NOT NULL,
                deadline   TIMESTAMPTZ NOT NULL,
                reminded   BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Bot 持久化：杂项状态（last_morning_date 等）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

    print("✅ 数据库表结构已就绪")


# ============================================================
# 中文分词工具
# ============================================================

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
EN_WORD_PATTERN = re.compile(r'[a-zA-Z0-9]+')
NUM_PATTERN = re.compile(r'\d{2,}')


def extract_search_keywords(query: str) -> List[str]:
    """
    从查询中提取搜索关键词
    
    中文：提取连续中文片段，拆成2字和3字词组（滑动窗口）
    英文：按空格分词
    数字：保留完整数字串（年份等）
    
    例如：
    "春节干了什么" → ["春节", "节干", "干了", "了什", "什么", "春节干", "节干了", "干了什", "了什么"]
    "Garan春节"   → ["Garan", "春节"]
    "2026除夕"    → ["2026", "除夕"]
    """
    keywords = set()
    
    for match in EN_WORD_PATTERN.finditer(query):
        word = match.group()
        if len(word) >= 2:
            keywords.add(word)
    
    for match in NUM_PATTERN.finditer(query):
        keywords.add(match.group())
    
    chinese_chars = []
    for char in query:
        if CJK_PATTERN.match(char):
            chinese_chars.append(char)
        else:
            if len(chinese_chars) >= 2:
                _add_chinese_ngrams(chinese_chars, keywords)
            chinese_chars = []
    if len(chinese_chars) >= 2:
        _add_chinese_ngrams(chinese_chars, keywords)
    
    return list(keywords)


def _add_chinese_ngrams(chars: List[str], keywords: set):
    """把连续中文字符拆成2字和3字词组"""
    text = "".join(chars)
    if len(text) <= 3:
        keywords.add(text)
    for i in range(len(text) - 1):
        keywords.add(text[i:i+2])
    if len(text) >= 3:
        for i in range(len(text) - 2):
            keywords.add(text[i:i+3])


# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
            session_id, role, content, model,
        )


async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
        return list(reversed(rows))


# ============================================================
# 记忆操作
# ============================================================

async def save_memory(content: str, importance: int = 5, source_session: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memories (content, importance, source_session) VALUES ($1, $2, $3)",
            content, importance, source_session,
        )


async def search_memories(query: str, limit: int = 10):
    """
    搜索相关记忆 —— 中文友好的加权搜索
    
    流程：
    1. 从查询中提取关键词（中文bigram/trigram + 英文单词 + 数字）
    2. 用 ILIKE 逐关键词匹配，统计命中数
    3. 加权排序：
       - 关键词命中率 * 0.5（命中越多越相关）
       - 重要程度    * 0.3（importance 1-10 归一化）
       - 崭新度      * 0.2（越新分越高，按天衰减）
    """
    keywords = extract_search_keywords(query)
    
    if not keywords:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 每个关键词命中得1分
        case_parts = []
        params = []
        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)
        
        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)
        
        # 至少命中一个关键词
        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = " OR ".join(where_parts)
        
        limit_idx = len(keywords) + 1
        params.append(limit)
        
        # 综合评分公式
        # recency: 今天≈1.0, 1天前≈0.5, 7天前≈0.125
        sql = f"""
            SELECT 
                id, content, importance, created_at,
                ({hit_count_expr}) AS hit_count,
                (
                    {WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0 +
                    {WEIGHT_IMPORTANCE} * importance::float / 10.0 +
                    {WEIGHT_RECENCY} * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0))
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, created_at DESC
            LIMIT ${limit_idx}
        """
        
        results = await conn.fetch(sql, *params)
        
        if results:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords)>8 else ''} → 命中 {len(results)} 条")
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (hits={r['hit_count']}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果")
        
        return results


async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, content, importance, created_at FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )


async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]


async def get_all_memories():
    """导出所有记忆（用于备份）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, importance, source_session, created_at FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]


async def get_all_memories_detail():
    """获取所有记忆（含 id，用于管理页面）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content, importance, source_session, created_at FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]


async def update_memory(memory_id: int, content: str = None, importance: int = None):
    """更新单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if content is not None and importance is not None:
            await conn.execute(
                "UPDATE memories SET content = $1, importance = $2 WHERE id = $3",
                content, importance, memory_id
            )
        elif content is not None:
            await conn.execute(
                "UPDATE memories SET content = $1 WHERE id = $2",
                content, memory_id
            )
        elif importance is not None:
            await conn.execute(
                "UPDATE memories SET importance = $1 WHERE id = $2",
                importance, memory_id
            )


async def delete_memory(memory_id: int):
    """删除单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def delete_memories_batch(memory_ids: list):
    """批量删除记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids
        )


# ============================================================
# Bot 持久化：对话历史
# ============================================================

async def append_bot_messages(messages: list):
    """追加新消息到 bot_conversation_history，并保留最近500条"""
    if not messages:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO bot_conversation_history (role, content) VALUES ($1, $2)",
            [(m["role"], m["content"]) for m in messages]
        )
        # 只保留最新500条
        await conn.execute("""
            DELETE FROM bot_conversation_history
            WHERE id NOT IN (
                SELECT id FROM bot_conversation_history ORDER BY id DESC LIMIT 500
            )
        """)


async def load_bot_conversation_history(limit: int = 500) -> list:
    """启动时从数据库恢复对话历史"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM bot_conversation_history ORDER BY id DESC LIMIT $1",
            limit
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ============================================================
# Bot 持久化：DDL 任务
# ============================================================

async def save_ddl_task(title: str, deadline) -> int:
    """保存一条 DDL 任务，返回数据库 id"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO bot_ddl_list (title, deadline) VALUES ($1, $2) RETURNING id",
            title, deadline
        )
        return row["id"]


async def load_ddl_tasks() -> list:
    """启动时加载未过期的 DDL 任务"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, deadline, reminded FROM bot_ddl_list WHERE deadline > NOW() ORDER BY deadline ASC"
        )
        return [{"db_id": r["id"], "title": r["title"], "deadline": r["deadline"], "reminded": r["reminded"]} for r in rows]


async def mark_ddl_reminded(db_id: int):
    """标记某条 DDL 已提醒"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE bot_ddl_list SET reminded = TRUE WHERE id = $1", db_id)


# ============================================================
# Bot 持久化：杂项状态
# ============================================================

async def set_bot_state(key: str, value: str):
    """存储任意状态值（upsert）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO bot_state (key, value, updated_at) VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
        """, key, value)


async def get_bot_state(key: str) -> str:
    """读取状态值，不存在返回 None"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM bot_state WHERE key = $1", key)
        return row["value"] if row else None
