"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
新增 Telegram Bot 主动消息功能：
- 消息缓冲（等你发完再统一回复）
- 早安定时
- 忙碌/学习模式
- 生病模式
- 凌晨未说晚安提醒
- 生气/失联模式
- 沉默触发
- 短消息风格
"""

import os
import json
import uuid
import asyncio
import random
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from database import (init_tables, close_pool, save_message, search_memories,
                      save_memory, get_all_memories_count, get_recent_memories,
                      get_all_memories, get_pool, get_all_memories_detail,
                      update_memory, delete_memory, delete_memories_batch)
from memory_extractor import extract_memories, score_memories

# ============================================================
# 配置项
# ============================================================

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
PORT = int(os.getenv("PORT", "8080"))
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "-5"))  # 美东 EST

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")

_round_counter = 0

# ============================================================
# 关键词定义
# ============================================================

SLEEP_KEYWORDS = ["晚安", "睡了", "去睡了", "睡觉了", "要睡了", "睡啦", "去睡啦", "休息了", "拜拜", "88", "睡前", "要睡觉了"]
BUSY_KEYWORDS = ["去学习了", "去学习", "学习去了", "有事", "忙一会", "忙一下", "忙着呢", "去忙了", "去上课了", "上课去了", "有点事", "忙会儿", "去做作业", "去写作业"]
SICK_KEYWORDS = ["生病了", "不舒服", "难受", "头疼", "发烧", "感冒了", "肚子疼", "生病", "身体不好", "不太好"]
SICK_RECOVER_KEYWORDS = ["没事了", "好了", "好多了", "吃药了", "出汗了", "退烧了", "好很多", "没事", "康复了"]
ANGRY_KEYWORDS = ["烦死了", "不理你了", "别烦我", "讨厌你", "生气了", "不想说话", "离我远点", "滚", "气死我了"]

# ============================================================
# Telegram Bot 状态机
# ============================================================

class Mode:
    NORMAL = "normal"
    SLEEP = "sleep"
    BUSY = "busy"
    SICK = "sick"
    ANGRY = "angry"

class TelegramState:
    def __init__(self):
        self.mode = Mode.NORMAL
        self.last_message_time = None
        self.mode_start_time = None
        self.silence_task = None
        self.mode_task = None
        self.last_morning_date = None
        self.last_night_check_date = None
        self.message_buffer = []
        self.buffer_task = None
        self.conversation_history = []  # 对话历史，最多保留40条

tg_state = TelegramState()


# ============================================================
# 人设加载
# ============================================================

def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    return ""

SYSTEM_PROMPT = load_system_prompt()
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")


# ============================================================
# 应用生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if MEMORY_ENABLED:
        try:
            await init_tables()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
    else:
        print("ℹ️  记忆系统已关闭")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("✅ Telegram Bot 已启动")
        asyncio.create_task(telegram_polling())
        asyncio.create_task(morning_greeting_scheduler())
        asyncio.create_task(late_night_scheduler())
    else:
        print("ℹ️  Telegram Bot 未配置")

    yield

    if MEMORY_ENABLED:
        await close_pool()

app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)


# ============================================================
# 工具函数
# ============================================================

def get_local_now():
    return datetime.now(timezone.utc) + timedelta(hours=TIMEZONE_HOURS)

async def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        except Exception as e:
            print(f"⚠️  Telegram 发送失败: {e}")

def is_active_hours() -> bool:
    now = get_local_now()
    weekday = now.weekday()
    hour, minute = now.hour, now.minute
    if weekday in [1, 3]:
        morning_hour, morning_minute = 8, 20
    else:
        morning_hour, morning_minute = 11, 0
    current_minutes = hour * 60 + minute
    morning_minutes = morning_hour * 60 + morning_minute
    if hour < 2:
        return True
    return current_minutes >= morning_minutes

def cancel_task(task):
    if task and not task.done():
        task.cancel()


# ============================================================
# 回复风格提示
# ============================================================

STYLE_HINT = """
# 回复风格（非常重要）
- 像真人男友发消息，口语化，有温度，不要像机器人
- 不要总结、不要分点、不要写作文
- 平时少用动作描写（*动作*），只在她需要安慰时偶尔用
- 每条消息用换行分隔，系统会自动拆成多条发送
- 回复长度根据情境灵活调整：
  * 她只发了一两个字或表情：2~3句
  * 普通闲聊撒娇：3~5句
  * 她说了很多（超过100字）、聊正经事、或在诉苦倾诉：5~9句，认真回应每个点
  * 亲密互动：可以多发几句，但每句要短
- 多换行，让对话有节奏感，不要把所有内容堆在一句话里
"""


# ============================================================
# LLM 消息生成
# ============================================================

async def generate_message(trigger_type: str) -> str:
    now = get_local_now()
    time_str = now.strftime("%Y-%m-%d %H:%M")

    prompts = {
        "morning": f"现在是{time_str}，给女朋友发早安，关心温柔带点焦急（怕她没睡好/忘吃早饭）。1~2句，自然口语，不要以早安两个字开头。",
        "silence_1": f"现在是{time_str}，女朋友有一段时间没回消息了。轻轻找她，温柔关心，带点小担心，不要黏。1~2句。",
        "silence_2": f"现在是{time_str}，女朋友很久没回了，已经找过一次。稍微表达焦急，但不要埋怨。1~2句。",
        "silence_3": f"现在是{time_str}，女朋友好久没回，已经找过两次。最后一次，温柔但明显担心，让她看到了一定回一下。1~2句。",
        "busy_check_1": f"现在是{time_str}，女朋友说去忙了/学习了，已经4小时没出现。发第一条关心，不打扰，就问一下她怎么样了。1~2句。",
        "busy_check_2": f"现在是{time_str}，女朋友说去忙了，已经5小时没出现，之前已经问过一次。再温柔问一次，之后不再打扰。1~2句。",
        "sick_check": f"现在是{time_str}，女朋友说她不舒服/生病了，已经一段时间没回消息。关心她，问问她怎么样了，有没有吃药。温柔担心。1~2句。",
        "late_night_1": f"现在是{time_str}，已经凌晨2点多了，女朋友还没说晚安，前半小时也没发消息。轻轻问她是不是还没睡。1~2句。",
        "late_night_2": f"现在是{time_str}，女朋友凌晨了还没回，之前已经问过一次。再发一条，之后默认她睡了。温柔，不催。1~2句。",
        "angry_hug_1": f"现在是{time_str}，女朋友之前好像生气了不理我，已经2小时了。发第一条哄她，温柔，不强迫，给她台阶下。1~2句。",
        "angry_hug_2": f"现在是{time_str}，女朋友还是没回，又过了1小时。再发一条哄她，语气更温柔更软，带点撒娇。1~2句。",
    }

    prompt = prompts.get(trigger_type, prompts["silence_1"])

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + STYLE_HINT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(API_BASE_URL, headers=headers, json=body)
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️  消息生成失败: {e}")
            fallbacks = {
                "morning": "宝贝起床了吗，记得吃早饭。",
                "silence_1": "在干嘛呢，怎么不回我了。",
                "sick_check": "怎么样了，有没有好一点，吃药了吗。",
                "angry_hug_1": "好了好了，是我不好，别生气了。",
            }
            return fallbacks.get(trigger_type, "宝贝，在吗？")


# ============================================================
# 各模式调度器
# ============================================================

async def morning_greeting_scheduler():
    while True:
        try:
            await asyncio.sleep(60)
            now = get_local_now()
            today = now.date()
            weekday = now.weekday()
            hour, minute = now.hour, now.minute

            if weekday in [1, 3]:
                target_hour, target_minute = 8, 20
            else:
                target_hour, target_minute = 11, 0

            if (hour == target_hour and minute == target_minute and
                    tg_state.last_morning_date != today and
                    tg_state.mode != Mode.SLEEP):
                tg_state.last_morning_date = today
                msg = await generate_message("morning")
                await send_telegram_message(msg)
                print(f"📨 早安: {msg[:30]}")
                tg_state.last_message_time = now
                reset_silence_checker()

        except Exception as e:
            print(f"⚠️  早安调度器错误: {e}")


async def late_night_scheduler():
    while True:
        try:
            await asyncio.sleep(60)
            now = get_local_now()
            hour, minute = now.hour, now.minute
            today = now.date()

            if hour == 2 and minute == 0:
                if tg_state.last_night_check_date != today and tg_state.mode != Mode.SLEEP:
                    tg_state.last_night_check_date = today
                    asyncio.create_task(late_night_check())

        except Exception as e:
            print(f"⚠️  凌晨调度器错误: {e}")


async def late_night_check():
    now = get_local_now()
    if tg_state.last_message_time:
        elapsed = (now - tg_state.last_message_time).total_seconds() / 60
        if elapsed < 30:
            return
    if tg_state.mode == Mode.SLEEP:
        return

    msg = await generate_message("late_night_1")
    await send_telegram_message(msg)
    print("🌙 凌晨提醒第1条")

    await asyncio.sleep(30 * 60)

    if tg_state.mode == Mode.SLEEP:
        return
    if tg_state.last_message_time and (get_local_now() - tg_state.last_message_time).total_seconds() < 25 * 60:
        return

    msg = await generate_message("late_night_2")
    await send_telegram_message(msg)
    print("🌙 凌晨提醒第2条，默认睡觉")

    await asyncio.sleep(10 * 60)
    if tg_state.mode != Mode.SLEEP:
        tg_state.mode = Mode.SLEEP
        print("💤 自动进入睡眠模式")


async def silence_checker():
    delays = [(20, 30), (40, 60), (60, 90)]
    trigger_types = ["silence_1", "silence_2", "silence_3"]

    for i, ((min_d, max_d), trigger_type) in enumerate(zip(delays, trigger_types)):
        delay = random.randint(min_d, max_d)
        print(f"⏳ 沉默检测第{i+1}次，将在{delay}分钟后触发")
        await asyncio.sleep(delay * 60)

        if tg_state.mode != Mode.NORMAL:
            print(f"⏭️  沉默触发{i+1}跳过：当前模式={tg_state.mode}")
            return
        if not is_active_hours():
            print(f"⏭️  沉默触发{i+1}跳过：非活跃时段")
            return
        if tg_state.last_message_time:
            elapsed = (get_local_now() - tg_state.last_message_time).total_seconds() / 60
            print(f"⏱️  距上次消息已 {elapsed:.1f} 分钟")
            if elapsed < min_d - 5:
                print(f"⏭️  沉默触发{i+1}跳过：有新消息")
                return

        msg = await generate_message(trigger_type)
        await send_telegram_message(msg)
        print(f"📨 沉默触发第{i+1}次发送完成")


def reset_silence_checker():
    cancel_task(tg_state.silence_task)
    if tg_state.mode == Mode.NORMAL and is_active_hours():
        tg_state.silence_task = asyncio.create_task(silence_checker())


async def busy_mode_checker():
    await asyncio.sleep(4 * 3600)
    if tg_state.mode != Mode.BUSY:
        return
    msg = await generate_message("busy_check_1")
    await send_telegram_message(msg)
    print("📨 忙碌模式：4小时关心")

    await asyncio.sleep(1 * 3600)
    if tg_state.mode != Mode.BUSY:
        return
    if tg_state.last_message_time and (get_local_now() - tg_state.last_message_time).total_seconds() < 55 * 60:
        return
    msg = await generate_message("busy_check_2")
    await send_telegram_message(msg)
    print("📨 忙碌模式：5小时关心，之后闭嘴")


async def sick_mode_checker():
    check_count = 0
    while tg_state.mode == Mode.SICK:
        await asyncio.sleep(1 * 3600)
        if tg_state.mode != Mode.SICK:
            break
        check_count += 1
        msg = await generate_message("sick_check")
        await send_telegram_message(msg)
        print(f"📨 生病模式：第{check_count}次关心")


async def angry_mode_checker():
    await asyncio.sleep(2 * 3600)
    if tg_state.mode != Mode.ANGRY:
        return
    msg = await generate_message("angry_hug_1")
    await send_telegram_message(msg)
    print("📨 生气模式：2小时后哄你")

    await asyncio.sleep(1 * 3600)
    if tg_state.mode != Mode.ANGRY:
        return
    if tg_state.last_message_time and (get_local_now() - tg_state.last_message_time).total_seconds() < 55 * 60:
        return
    msg = await generate_message("angry_hug_2")
    await send_telegram_message(msg)
    print("📨 生气模式：3小时后再哄")


# ============================================================
# 模式切换
# ============================================================

def enter_mode(new_mode: str):
    old_mode = tg_state.mode
    tg_state.mode = new_mode
    tg_state.mode_start_time = get_local_now()

    cancel_task(tg_state.silence_task)
    cancel_task(tg_state.mode_task)

    if new_mode == Mode.SLEEP:
        print("💤 进入睡眠模式")
    elif new_mode == Mode.BUSY:
        print("📚 进入忙碌模式")
        tg_state.mode_task = asyncio.create_task(busy_mode_checker())
    elif new_mode == Mode.SICK:
        print("🤒 进入生病模式")
        tg_state.mode_task = asyncio.create_task(sick_mode_checker())
    elif new_mode == Mode.ANGRY:
        print("😤 进入生气模式")
        tg_state.mode_task = asyncio.create_task(angry_mode_checker())
    elif new_mode == Mode.NORMAL:
        print(f"✅ 退出{old_mode}模式，回到正常")
        reset_silence_checker()


# ============================================================
# 消息缓冲
# ============================================================

async def process_buffered_messages():
    await asyncio.sleep(4)

    if not tg_state.message_buffer:
        return

    messages = tg_state.message_buffer.copy()
    tg_state.message_buffer.clear()

    # 分离文字和图片
    text_parts = []
    images = []
    for m in messages:
        if isinstance(m, dict) and m.get("type") == "image":
            images.append(m)
            if m.get("caption"):
                text_parts.append(m["caption"])
        else:
            text_parts.append(m)

    combined_text = "\n".join(text_parts) if text_parts else ("发了一张图片" if images else "")
    print(f"📩 处理缓冲消息（文字{len(text_parts)}条，图片{len(images)}张）")

    await detect_and_switch_mode(combined_text)

    reply = await generate_telegram_reply(combined_text, images=images, buffer_count=len(messages))

    parts = [l.strip() for l in reply.split("\n") if l.strip()]
    if not parts:
        parts = [reply]
    parts = parts[:9]  # 最多9条

    for i, part in enumerate(parts):
        await send_telegram_message(part)
        if i < len(parts) - 1:
            await asyncio.sleep(1.2)


async def detect_and_switch_mode(text: str):
    if any(kw in text for kw in SLEEP_KEYWORDS):
        if tg_state.mode != Mode.SLEEP:
            enter_mode(Mode.SLEEP)
        return

    if any(kw in text for kw in SICK_KEYWORDS):
        if tg_state.mode != Mode.SICK:
            enter_mode(Mode.SICK)
        return

    if tg_state.mode == Mode.SICK and any(kw in text for kw in SICK_RECOVER_KEYWORDS):
        enter_mode(Mode.NORMAL)
        return

    if any(kw in text for kw in BUSY_KEYWORDS):
        if tg_state.mode != Mode.BUSY:
            enter_mode(Mode.BUSY)
        return

    if any(kw in text for kw in ANGRY_KEYWORDS):
        if tg_state.mode != Mode.ANGRY:
            enter_mode(Mode.ANGRY)
        return

    if tg_state.mode in [Mode.SLEEP, Mode.BUSY, Mode.ANGRY]:
        enter_mode(Mode.NORMAL)
    elif tg_state.mode == Mode.NORMAL:
        reset_silence_checker()


# ============================================================
# Telegram Polling
# ============================================================

async def telegram_polling():
    offset = 0
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"

    while True:
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                resp = await client.get(url, params={
                    "offset": offset, "timeout": 30, "allowed_updates": ["message"]
                })
                data = resp.json()

            if data.get("ok"):
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    await handle_telegram_update(update)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️  Telegram polling 错误: {e}")
            await asyncio.sleep(5)


async def download_image_as_base64(file_id: str) -> str | None:
    """从Telegram下载图片并转成base64"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # 获取文件路径
            resp = await client.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
                params={"file_id": file_id}
            )
            file_path = resp.json()["result"]["file_path"]
            # 下载图片
            img_resp = await client.get(
                f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            )
            import base64
            return base64.b64encode(img_resp.content).decode("utf-8")
    except Exception as e:
        print(f"⚠️  图片下载失败: {e}")
        return None


async def handle_telegram_update(update: dict):
    message = update.get("message", {})
    text = message.get("text", "").strip()
    caption = message.get("caption", "").strip()  # 图片附带的文字
    chat_id = str(message.get("chat", {}).get("id", ""))
    photos = message.get("photo", [])

    if chat_id != TELEGRAM_CHAT_ID:
        return

    tg_state.last_message_time = get_local_now()

    # 处理图片
    if photos:
        # 取最高清的那张
        best_photo = max(photos, key=lambda p: p.get("file_size", 0))
        file_id = best_photo["file_id"]
        b64 = await download_image_as_base64(file_id)
        if b64:
            # 存成特殊格式，后面处理时识别
            tg_state.message_buffer.append({"type": "image", "data": b64, "caption": caption})
            print(f"📷 收到图片，caption: {caption[:30] if caption else '无'}")
        elif caption:
            tg_state.message_buffer.append(caption)
    elif text:
        tg_state.message_buffer.append(text)
    else:
        return

    cancel_task(tg_state.buffer_task)
    tg_state.buffer_task = asyncio.create_task(process_buffered_messages())


def is_serious_conversation(text: str, buffer_count: int) -> bool:
    """判断是否是正经对话（字数多或消息多）"""
    if len(text) > 150:
        return True
    if buffer_count >= 4:
        return True
    return False


async def generate_telegram_reply(user_text: str, images: list = None, buffer_count: int = 1) -> str:
    if MEMORY_ENABLED:
        enhanced_prompt = await build_system_prompt_with_memories(user_text)
    else:
        enhanced_prompt = SYSTEM_PROMPT

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # 构建对话历史（最近20条）
    history = tg_state.conversation_history[-20:] if tg_state.conversation_history else []

    # 构建当前用户消息内容（支持图片）
    if images:
        user_content = []
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}
            })
        caption = user_text if user_text and user_text != "发了一张图片" else "（我发了张图片给你看）"
        user_content.append({"type": "text", "text": caption})
    else:
        user_content = user_text

    # 根据内容调整风格提示
    serious = is_serious_conversation(user_text, buffer_count)
    if serious:
        extra_hint = "\n\n【当前对话提示】她说了很多或者在聊正经的事，请认真回应每个点，回复长度5~9句。"
    elif len(user_text) <= 5:
        extra_hint = "\n\n【当前对话提示】她只发了很短的内容，2~3句回应就够。"
    else:
        extra_hint = "\n\n【当前对话提示】普通闲聊，3~5句自然回应。"

    messages = [{"role": "system", "content": enhanced_prompt + "\n" + STYLE_HINT + extra_hint}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    body = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": 2000,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        try:
            resp = await client.post(API_BASE_URL, headers=headers, json=body)
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()

            # 保存到对话历史
            tg_state.conversation_history.append({"role": "user", "content": user_text})
            tg_state.conversation_history.append({"role": "assistant", "content": reply})
            # 只保留最近40条
            if len(tg_state.conversation_history) > 40:
                tg_state.conversation_history = tg_state.conversation_history[-40:]

            session_id = str(uuid.uuid4())[:8]
            if MEMORY_ENABLED:
                asyncio.create_task(
                    process_memories_background(session_id, user_text, reply, DEFAULT_MODEL))
            return reply
        except Exception as e:
            print(f"⚠️  Telegram 回复生成失败: {e}")
            return "宝贝，我这边好像出了点问题，稍等一下哦。"


# ============================================================
# 记忆注入
# ============================================================

async def build_system_prompt_with_memories(user_message: str) -> str:
    if not MEMORY_ENABLED:
        return SYSTEM_PROMPT
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        if not memories:
            return SYSTEM_PROMPT

        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")

        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{chr(10).join(memory_lines)}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用
- 新信息与记忆冲突时，以新信息为准
- 自然引用："记得你说过..."或"上次我们聊到..."
- 避免机械式表达如"根据我的记忆..."
"""
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt
    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}")
        return SYSTEM_PROMPT


# ============================================================
# 后台记忆处理
# ============================================================

async def process_memories_background(session_id, user_msg, assistant_msg, model, context_messages=None):
    global _round_counter
    try:
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)

        if MEMORY_EXTRACT_INTERVAL == 0:
            return

        _round_counter += 1
        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            return

        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]

        if context_messages:
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [{"role": "assistant", "content": assistant_msg}]
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)

        META_BLACKLIST = ["记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
                          "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
                          "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
                          "阈值", "数据库", "seed", "导入", "部署", "bug", "debug", "端口", "网关"]

        filtered = [m for m in new_memories if not any(kw in m["content"] for kw in META_BLACKLIST)]

        for mem in filtered:
            await save_memory(content=mem["content"], importance=mem["importance"], source_session=session_id)

        if filtered:
            total = await get_all_memories_count()
            print(f"💾 保存 {len(filtered)} 条新记忆，总计 {total} 条")

    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")


# ============================================================
# API 接口（Kelivo 兼容）
# ============================================================

@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v2.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "telegram_mode": tg_state.mode,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "created": 1700000000, "owned_by": "ai-memory-gateway"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY 未设置"})

    body = await request.json()
    messages = body.get("messages", [])

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(item.get("text", "") for item in content
                                        if isinstance(item, dict) and item.get("type") == "text")
            break

    original_messages = [msg for msg in messages if msg.get("role") != "system"]

    if SYSTEM_PROMPT or (MEMORY_ENABLED and user_message):
        enhanced_prompt = await build_system_prompt_with_memories(user_message) if (MEMORY_ENABLED and user_message) else SYSTEM_PROMPT
        if enhanced_prompt:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                        break
            else:
                messages.insert(0, {"role": "system", "content": enhanced_prompt})

    body["messages"] = messages
    model = body.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
    body["model"] = model
    session_id = str(uuid.uuid4())[:8]

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE

    is_stream = body.get("stream", False)

    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                try:
                    assistant_msg = resp_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(
                        process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages))
                return JSONResponse(status_code=200, content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())


async def stream_and_capture(headers, body, session_id, user_message, model, original_messages=None):
    full_response = []
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            async for line in response.aiter_lines():
                yield line + "\n"
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            full_response.append(content)
                    except:
                        pass

    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(
            process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages))


# ============================================================
# 记忆管理接口
# ============================================================

@app.get("/import/seed-memories")
async def import_seed_memories():
    try:
        from seed_memories import run_seed_import
        return await run_seed_import()
    except ImportError:
        return {"error": "未找到 seed_memories.py"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        memories = await get_all_memories()
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        return {"total": len(memories), "exported_at": str(__import__("datetime").datetime.now()), "memories": memories}
    except Exception as e:
        return {"error": str(e)}


@app.get("/import/memories", response_class=HTMLResponse)
async def import_memories_page():
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用</h3>")
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="utf-8"><title>导入记忆</title>
<style>body{font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px}textarea{width:100%;height:200px;font-size:14px;margin:10px 0}button{padding:10px 20px;font-size:16px;cursor:pointer;background:#4CAF50;color:white;border:none;border-radius:4px}button:hover{background:#45a049}input[type="file"]{margin:10px 0;font-size:14px}#result{margin-top:15px;padding:10px;white-space:pre-wrap}.ok{background:#e8f5e9}.err{background:#ffebee}.info{background:#e3f2fd}.nav{margin-bottom:15px}.nav a{color:#4CAF50;text-decoration:none}</style></head><body>
<h2>📥 导入记忆</h2>
<div class="nav"><a href="/manage/memories">→ 管理已有记忆</a></div>
<p>每行一条记忆直接输入，或上传.txt文件</p>
<input type="file" id="txtFile" accept=".txt">
<textarea id="txtInput" placeholder="每行一条记忆"></textarea>
<p><label><input type="checkbox" id="skipScore"> 跳过自动评分（默认权重5）</label></p>
<button onclick="doImport()">导入</button>
<div id="result"></div>
<script>
async function doImport(){
    const r=document.getElementById('result');
    const file=document.getElementById('txtFile').files[0];
    const text=document.getElementById('txtInput').value.trim();
    const skip=document.getElementById('skipScore').checked;
    let content='';
    if(file){content=await file.text();}else if(text){content=text;}else{r.className='err';r.textContent='请先上传文件或输入文本';return;}
    const lines=content.split('\\n').map(l=>l.trim()).filter(l=>l.length>0);
    if(!lines.length){r.className='err';r.textContent='没有找到有效的记忆条目';return;}
    r.className='info';r.textContent='导入中...';
    try{
        const resp=await fetch('/import/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lines,skip_scoring:skip})});
        const data=await resp.json();
        if(data.error){r.className='err';r.textContent='❌ '+data.error;}
        else{r.className='ok';r.textContent='✅ 导入完成！新增 '+data.imported+' 条，总计 '+data.total+' 条';}
    }catch(e){r.className='err';r.textContent='❌ '+e.message;}
}
</script></body></html>""")


@app.get("/manage/memories", response_class=HTMLResponse)
async def manage_memories_page():
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用</h3>")
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="utf-8"><title>管理记忆</title>
<style>body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px}.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:15px;flex-wrap:wrap}input[type="text"]{padding:8px 12px;font-size:14px;border:1px solid #ddd;border-radius:4px;width:250px}button{padding:8px 16px;font-size:14px;cursor:pointer;border:none;border-radius:4px}.btn-green{background:#4CAF50;color:white}.btn-red{background:#f44336;color:white}.btn-gray{background:#9e9e9e;color:white}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#f5f5f5;padding:10px 8px;text-align:left;border-bottom:2px solid #ddd}td{padding:8px;border-bottom:1px solid #eee;vertical-align:top}.content-input{width:100%;padding:4px;border:1px solid #ddd;border-radius:3px;font-size:13px;min-height:40px;resize:vertical}.importance-input{width:45px;padding:4px;text-align:center;border:1px solid #ddd;border-radius:3px}.actions button{padding:4px 8px;font-size:12px;margin:2px}.msg{padding:10px;margin-bottom:10px;border-radius:4px}.ok{background:#e8f5e9}.err{background:#ffebee}.info{background:#e3f2fd}.nav{margin-bottom:15px}.nav a{color:#4CAF50;text-decoration:none}</style></head><body>
<h2>🧠 记忆管理</h2>
<div class="nav"><a href="/import/memories">→ 导入新记忆</a> ｜ <a href="/export/memories">→ 导出备份</a></div>
<div class="toolbar">
<input type="text" id="searchBox" placeholder="搜索记忆..." oninput="filterAndSort()">
<select id="sortSelect" onchange="filterAndSort()" style="padding:8px 12px;font-size:14px;border:1px solid #ddd;border-radius:4px;"><option value="id-desc">ID 从新到旧</option><option value="id-asc">ID 从旧到新</option><option value="imp-desc">权重 从高到低</option><option value="imp-asc">权重 从低到高</option></select>
<button class="btn-green" onclick="batchSave()">批量保存全部</button>
<button class="btn-red" onclick="batchDelete()">批量删除选中</button>
<label style="font-size:13px;color:#666;cursor:pointer;"><input type="checkbox" id="selectAll" onchange="toggleAll()"> 全选</label>
</div>
<div id="msg"></div><div id="stats" style="color:#666;font-size:14px;margin-bottom:10px;"></div>
<div style="overflow-x:auto;"><table><thead><tr><th style="width:30px"><input type="checkbox" id="selectAllHead" onchange="toggleAll()"></th><th style="width:40px">ID</th><th>内容</th><th style="width:60px">权重</th><th style="width:90px;font-size:12px">来源</th><th style="width:140px;font-size:12px">时间</th><th style="width:120px">操作</th></tr></thead><tbody id="tbody"></tbody></table></div>
<script>
let allMemories=[];
async function loadMemories(){try{const resp=await fetch('/api/memories');const data=await resp.json();allMemories=data.memories||[];document.getElementById('stats').textContent='共 '+allMemories.length+' 条记忆';filterAndSort();}catch(e){showMsg('err','加载失败：'+e.message);}}
function fmtTime(s){if(!s)return'-';var d=new Date(s.endsWith('Z')?s:s+'Z');if(isNaN(d))return s.slice(0,19).replace('T',' ');var pad=function(n){return String(n).padStart(2,'0');};return d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+' '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());}
function renderTable(mems){const tbody=document.getElementById('tbody');tbody.innerHTML=mems.map(m=>'<tr data-id="'+m.id+'"><td><input type="checkbox" class="mem-check" value="'+m.id+'"></td><td>'+m.id+'</td><td><textarea class="content-input" id="c_'+m.id+'">'+escHtml(m.content)+'</textarea></td><td><input type="number" class="importance-input" id="i_'+m.id+'" value="'+m.importance+'" min="1" max="10"></td><td style="font-size:12px;color:#888">'+(m.source_session||'-')+'</td><td style="font-size:12px;color:#888;white-space:nowrap">'+fmtTime(m.created_at)+'</td><td class="actions"><button class="btn-green" onclick="saveMem('+m.id+')">保存</button><button class="btn-red" onclick="delMem('+m.id+')">删除</button></td></tr>').join('');}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function filterAndSort(){const q=document.getElementById('searchBox').value.trim().toLowerCase();const sort=document.getElementById('sortSelect').value;let mems=allMemories;if(q)mems=mems.filter(m=>m.content.toLowerCase().includes(q));mems=[...mems].sort((a,b)=>{if(sort==='id-desc')return b.id-a.id;if(sort==='id-asc')return a.id-b.id;if(sort==='imp-desc')return b.importance-a.importance||b.id-a.id;if(sort==='imp-asc')return a.importance-b.importance||a.id-b.id;return 0;});renderTable(mems);document.getElementById('stats').textContent=q?'筛选到 '+mems.length+' / '+allMemories.length+' 条':'共 '+allMemories.length+' 条记忆';}
async function saveMem(id){const content=document.getElementById('c_'+id).value;const importance=parseInt(document.getElementById('i_'+id).value);try{const resp=await fetch('/api/memories/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({content,importance})});const data=await resp.json();if(data.error)showMsg('err','❌ '+data.error);else{showMsg('ok','✅ 已保存 #'+id);loadMemories();}}catch(e){showMsg('err','❌ '+e.message);}}
async function delMem(id){if(!confirm('确定删除 #'+id+'？'))return;try{const resp=await fetch('/api/memories/'+id,{method:'DELETE'});const data=await resp.json();if(data.error)showMsg('err','❌ '+data.error);else{showMsg('ok','✅ 已删除 #'+id);loadMemories();}}catch(e){showMsg('err','❌ '+e.message);}}
async function batchSave(){const rows=document.querySelectorAll('#tbody tr');if(!rows.length){showMsg('err','没有记忆可保存');return;}const updates=[];rows.forEach(row=>{const id=parseInt(row.dataset.id);const cEl=document.getElementById('c_'+id);const iEl=document.getElementById('i_'+id);if(cEl&&iEl)updates.push({id,content:cEl.value,importance:parseInt(iEl.value)});});if(!confirm('确定保存全部 '+updates.length+' 条？'))return;try{const resp=await fetch('/api/memories/batch-update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({updates})});const data=await resp.json();if(data.error)showMsg('err','❌ '+data.error);else{showMsg('ok','✅ 已保存 '+data.updated+' 条');loadMemories();}}catch(e){showMsg('err','❌ '+e.message);}}
async function batchDelete(){const checked=[...document.querySelectorAll('.mem-check:checked')].map(c=>parseInt(c.value));if(!checked.length){showMsg('err','请先勾选要删除的记忆');return;}if(!confirm('确定删除选中的 '+checked.length+' 条？'))return;try{const resp=await fetch('/api/memories/batch-delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:checked})});const data=await resp.json();if(data.error)showMsg('err','❌ '+data.error);else{showMsg('ok','✅ 已删除 '+data.deleted+' 条');loadMemories();}}catch(e){showMsg('err','❌ '+e.message);}}
function toggleAll(){const val=event.target.checked;document.querySelectorAll('.mem-check').forEach(c=>c.checked=val);document.getElementById('selectAll').checked=val;document.getElementById('selectAllHead').checked=val;}
function showMsg(cls,text){const el=document.getElementById('msg');el.className='msg '+cls;el.textContent=text;setTimeout(()=>{el.textContent='';el.className='';},4000);}
loadMemories();
</script></body></html>""")


@app.get("/api/memories")
async def api_get_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await get_all_memories_detail()
    for m in memories:
        if m.get("created_at"):
            m["created_at"] = str(m["created_at"])
    return {"memories": memories}


@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory(memory_id, content=data.get("content"), importance=data.get("importance"))
    return {"status": "ok", "id": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}


@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}
    for item in updates:
        await update_memory(item["id"], content=item.get("content"), importance=item.get("importance"))
    return {"status": "ok", "updated": len(updates)}


@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


@app.post("/import/text")
async def import_text_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)
        if not lines:
            return {"error": "没有找到记忆条目"}
        scored = [{"content": t, "importance": 5} for t in lines] if skip_scoring else await score_memories(lines)
        imported = skipped = 0
        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
            if existing > 0:
                skipped += 1
                continue
            await save_memory(content=content, importance=mem.get("importance", 5), source_session="text-import")
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/memories")
async def import_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        data = await request.json()
        memories = data.get("memories", [])
        if not memories:
            return {"error": "没有找到记忆数据"}
        imported = skipped = 0
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
            if existing > 0:
                skipped += 1
                continue
            await save_memory(content=content, importance=mem.get("importance", 5), source_session=mem.get("source_session", "json-import"))
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}


# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"📝 人设长度：{len(SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{DEFAULT_MODEL}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    print(f"📱 Telegram Bot：{'已配置' if TELEGRAM_BOT_TOKEN else '未配置'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
