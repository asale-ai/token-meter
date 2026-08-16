# token-meter

[English](README.md) | 简体中文

跨厂商的 LLM token 计量与费用核算库,Rust 实现,覆盖 **Claude**、**GPT 系列**
和 **Gemini**。

发请求前先估算 prompt。拿到响应后读厂商真正计了多少。两者都能算成钱。然后把它们
放在一起比对。

每个计数都带着自己的出处,所以你永远知道手里拿的是实测值还是估算值。

```rust
use token_meter::{Prompt, Message, Content, Source};

let msgs = [Message::user([Content::Text("一句话解释 BPE。")])];
let count = Prompt::new("claude-sonnet-5")
    .system("回答要简短。")
    .messages(&msgs)
    .count();

println!("{} tokens ({:?})", count.tokens, count.source);
```

## 为什么每个计数都要带出处

一个不说明来源的 token 数是没法安全使用的。tiktoken 的精确值和字符类估算的
±10% 结果,在拿去和厂商账单比对时该配完全不同的容差 —— 一视同仁的代码要么过度
信任估算值,要么白白浪费了精确值。

```rust
if count.source.is_precise() {
    // tiktoken 或厂商自己的接口 —— 可以用紧阈值比对
} else {
    // 估算值 —— 留出余量
}
```

`Source` 按可信度从弱到强排序,总和会降级到最弱的那个输入:prompt 里任何一部分
是估算的,整个总数就是估算的。

## 能算什么

| | Claude | GPT 系列 | Gemini |
|---|---|---|---|
| 文本 | 启发式 | **精确**(`openai-exact`) | 启发式 |
| tool 定义 | JSON 估算 | **TypeScript 声明** | JSON 估算 |
| tool 调用与结果 | ✓ | ✓ | ✓ |
| 回放的思维链 | ✓ | ✓ | ✓ |
| 图片 | 面积 ÷ 750 | 512px 分块 | 768px 分块 |
| 文档 | 按体积 | 按体积 | 按体积 |
| 消息框架开销 | ✓ | ✓ | ✓ |

**tool 定义不是按 JSON 算的。** GPT 系列会把你的 tool 列表改写成一段 TypeScript
namespace 声明再送进模型,真正计费的是这段声明:

```text
namespace functions {

// Get the weather in a location
type get_weather = (_: {
// The city and state
location: string,
unit?: "celsius" | "fahrenheit",
}) => any;

} // namespace functions
```

按原始 JSON 计数会高估 agentic prompt 里最大的那块固定成本,而且是**每一轮都
高估**。`token_meter::tools::format_definitions` 会把真正被计数的字符串渲染出来,
方便你查看、diff 和写测试。

**图片是从文件头读真实尺寸的。** PNG、JPEG、GIF、WebP 的宽高直接解析头部 ——
不解码图像,不引入任何依赖 —— 因为一张 1024×1024 的截图在 Claude 上是一千多
token,不是朴素估算器假设的固定 85。头部读不出来时退回固定值,而不是凭空编一个
数字。

## 读取真实计费用量

```rust
use token_meter::Usage;

let usage = Usage::from_response(&frame);   // 任意方言,任意嵌套层级
```

这里有两个陷阱,都会悄无声息地把数字算错:

**prompt 总数 vs prompt 余量。** Anthropic 的 `input_tokens` **不含**缓存部分,
而 OpenAI 的 `prompt_tokens` 和 Gemini 的 `promptTokenCount` **含**。把两者映射到
同一个字段,缓存那部分就会被计两次 —— 单价虽只有十分之一,但它占 agentic prompt
的大头。这个库靠响应里出现了哪些明细字段来判断是哪种约定,而不是靠缓存数是否为零。

**Gemini 的思考不在 candidates 里。** `candidatesTokenCount` 只数可见回答,
`thoughtsTokenCount` 是并列字段,按 output 价计费。只读前者会漏报每一个推理轮次,
有时漏掉的是大半个 turn。响应里带 `totalTokenCount` 时,会先用它确认两者不重叠再
相加。

流式同理 —— `merge_response` 会把多个帧折叠进同一个累计值,因为 Anthropic 把
prompt 侧放在 `message_start`、output 侧放在 `message_delta`。

## 计费

```rust
use token_meter::{Rates, RateCard};

// 每百万 token:输入 $3、输出 $15、缓存读 $0.30、缓存写 $3.75,单位 micro-USD。
let rates = Rates::per_million(3_000_000, 15_000_000, 300_000, 3_750_000);
let cost = rates.cost(&usage);
```

费率是整数,单位为"每百万 token 的最小货币单位",中间运算走 `i128` —— 过了一遍
浮点的钱就是对不上账的钱。也提供了价格表常见约定的构造器(`per_1k`、`per_token`)。

`RateCard` 支持长上下文分层,而且是**全有全无**的:请求一旦越过阈值,整个请求重新
按高档计价,output 也一起,而不是只对超出的那部分加价。

`Cost::without_cache` 给出"完全不用缓存会花多少"的对照值。这个值两个方向都值得
盯:在对缓存写入计费的厂商上,一个总是命不中的缓存**比不用缓存更贵**。

## 比对

```rust
use token_meter::{compare, Policy, Source};

let dev = compare(estimated_input, observed_output, &reported, Policy::for_source(estimated_input.source));
```

`Policy::for_source` 会按你的计数方式选阈值 —— tiktoken 的结果可以卡到几个百分点,
字符类估算不行。判定默认是单向的(`Direction::OverOnly`),而且除了比例还要求绝对
token 差达标:短请求上只看比例毫无意义,会把每一笔诚实的小请求都变成告警。

`check_split` 是总量比对做不到的那个检查。把 prompt token 报成 `cache_write` 而不是
`cache_read`,所有总量都分毫不差,账单却翻十倍以上:

```rust
let finding = check_split(&predicted, &reported, &rates, min_cost);
```

它的阈值是**钱**而不是 token,因为这个检查本来关心的就是钱。

## 不做什么

**不预测输出 token。** 模型写出来之前,没有任何方法能知道答案有多长。
`StreamMeter` 是在流经过时做测量,是事后计数而不是预测。而且在计费推理 token 却
不流式下发的 wire 上(OpenAI Responses、Gemini),连这个测量都结构性地少于厂商实
际收费 —— `Dialect::output_estimate_multiple` 会告诉你少多少,这个倍率只用于比
对,**绝不能进计费**。

**不单独预测缓存命中。** 一段 prefix 会不会命中缓存,取决于厂商服务端的状态,本
地计算看不到。这个库算的是 prompt 的**可缓存范围**(`Prompt::count_prefix` 与
`Prompt::prefix_fingerprint`);命中历史由你提供 —— 走 `PrefixSeen` trait,或者如果
你的查询是异步的,直接把结果传给 `predict_seen`。得到的 `CacheSplit` 永远标记为
预测值。

用一个带 TTL 的存储(Redis,过期时间设成厂商的缓存窗口)托底,它可以用来估算开销,
也可以用来校验交易对手上报的切分 —— `cache_write` 通常是 1.25×、`cache_read` 是
0.1×,中间 12.5 倍的价差是只看总量的比对完全发现不了的。**但不要拿它计费**,而且在
把它接到任何会惩罚谁的机制之前,先拿真实结算数据量一遍误差:它天生会低估命中,因为
上游账号通常比任何单个调用方的历史所知道的更"热"。

**不内置价格表。** 算术它来做,费率你来给。费率随模型、地区、合同、日期而变,一个
声称知道你费率的库,就是一个会在钱上悄悄算错的库。

**不发明 Claude tokenizer。** Claude 3+ 的 BPE 从未公开。Anthropic 给的答案是
`/v1/messages/count_tokens`,这个库给的答案是 `RemoteCounter` trait —— 你把它接到
自己的 HTTP 客户端上,任何失败都会退回本地估算,返回值会告诉你实际走的哪条路。

## Features

```toml
[dependencies]
token-meter = { version = "0.1", features = ["openai-exact"] }
```

- **default** —— 除 `serde_json` 外无依赖,全部走估算。
- **`openai-exact`** —— 引入 `tiktoken-rs`,GPT 系列走真实分词。BPE 表有几 MB,
  以单例形式借用而不是克隆。

## 许可

Apache-2.0
