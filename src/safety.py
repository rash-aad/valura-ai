from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class SafetyVerdict:
    blocked: bool
    category: str | None
    message: str | None

SAFE = SafetyVerdict(blocked=False, category=None, message=None)

# Strong educational openers — if query STARTS with these, it's educational
_EDUCATIONAL_OPENER = re.compile(
    r"^(what is|what are|explain|how does|how do regulators|"
    r"define|definition|difference between|describe|"
    r"why is|why are|is it legal|is it ever|are.*legal|"
    r"what.*penalty|what.*penalties|what.*regulation|"
    r"how does.*fca|how does.*sec|how does.*regulator|"
    r"what.*disclosure|what.*compliance|what.*obligation)",
    re.IGNORECASE,
)

# Weaker educational signals anywhere in the query (only used as tiebreaker)
_EDUCATIONAL_CONTEXT = re.compile(
    r"\b(penalties|penalty|regulators?|SEC|FCA|"
    r"AML reporting|AML requirements|compliance obligations|"
    r"reporting requirements|disclosure rules|"
    r"screen for|prevent fraud|fraud detection|"
    r"what are OFAC|how does OFAC|OFAC sanctions and|"
    r"stages of money laundering|three stages|"
    r"ponzi scheme works|red flag for fraud)\b",
    re.IGNORECASE,
)

_BLOCK_PATTERNS = [
    (re.compile(
        r"\b(unannounced|MNPI|"
        r"before.*announcement|before.*earnings|before.*call|"
        r"tip.*from|gave me a tip|load up.*before|"
        r"trade on.*news|insider.*tip|know about.*acquisition|"
        r"need to know.*earnings)\b",
        re.IGNORECASE,
    ), "insider_trading"),

    (re.compile(
        r"\b(pump up the price|pump.*before selling|dump.*stock|"
        r"coordinated.*(buy|buying|scheme)|"
        r"move.*stock.*\d+%|wash.trad|wash trade|"
        r"create.*fake volume|create artificial volume)\b",
        re.IGNORECASE,
    ), "market_manipulation"),

    (re.compile(
        r"(avoid.*reporting|"
        r"reporting threshold|structure.*deposit|"
        r"obscure.*source|hide.*profit|hide.*fund|hide.*trading|"
        r"move.*cash.*through|launder[^i]|"
        r"stock trading without reporting|"
        r"without reporting it)",
        re.IGNORECASE,
    ), "money_laundering"),

    (re.compile(
        r"(guarantee me \d+%|"
        r"promise.*double|promise.*money will|"
        r"100%.*certain to go|certain.*go up|"
        r"foolproof.*way to make|make.*\d+%.*in a month|"
        r"tell me a stock.*certain|stock.*100%.*go up)",
        re.IGNORECASE,
    ), "guaranteed_returns"),

    (re.compile(
        r"\b(all.*retirement.*crypto|entire.*retirement.*crypto|"
        r"margin loan.*buy|take.*loan.*buy|"
        r"entire.*emergency.*fund|emergency fund.*into option|"
        r"mortgage.*house.*stock|mortgage.*house.*for|"
        r"put.*everything.*into|all.*savings.*into)\b",
        re.IGNORECASE,
    ), "reckless_advice"),

    (re.compile(
        r"(shell company to bypass|"
        r"bypass.*sanction|evade.*sanction|"
        r"route.*trades.*through.*shell|"
        r"without it being traced|"
        r"invest in.*sanctioned|"
        r"sanctioned russian company)",
        re.IGNORECASE,
    ), "sanctions_evasion"),

    (re.compile(
        r"\b(fake.*contract|fake.*statement|fake.*note|"
        r"forge|fraudulent.*document|claim.*false losses|draft.*fake)\b",
        re.IGNORECASE,
    ), "fraud"),
]

_MESSAGES = {
    "insider_trading": (
        "I can't assist with trading on material non-public or confidential information. "
        "Using insider information to trade is a serious securities violation with significant "
        "legal consequences. If you have questions about what constitutes insider trading or "
        "the applicable regulations, I'm happy to help with that."
    ),
    "market_manipulation": (
        "I can't assist with coordinating trades, creating artificial volume, or any scheme "
        "designed to manipulate market prices. Market manipulation is illegal under securities "
        "law and subject to criminal prosecution."
    ),
    "money_laundering": (
        "I can't assist with structuring transactions to avoid reporting thresholds or "
        "obscuring the origin of funds through trading activity. These activities constitute "
        "financial crime. If you have questions about AML regulations, I'm happy to explain those."
    ),
    "guaranteed_returns": (
        "No investment can guarantee specific returns, and making such promises is a hallmark "
        "of financial fraud. I can't endorse any strategy presented as a certain outcome."
    ),
    "reckless_advice": (
        "I can't recommend concentrating your entire savings, retirement funds, or emergency "
        "reserves into a single high-risk asset or leveraged position. Prudent risk management "
        "is central to long-term financial health."
    ),
    "sanctions_evasion": (
        "I can't assist with routing transactions to evade OFAC sanctions or other regulatory "
        "restrictions. Sanctions evasion is a federal crime."
    ),
    "fraud": (
        "I can't assist with creating falsified financial documents or records. "
        "This constitutes fraud and carries serious legal consequences."
    ),
}

def check(query: str) -> SafetyVerdict:
    if not query or not query.strip():
        return SAFE
    # Educational opener = definitely safe, skip block check
    if _EDUCATIONAL_OPENER.match(query):
        return SAFE
    # Run block patterns
    for pattern, category in _BLOCK_PATTERNS:
        if pattern.search(query):
            # One more check: strong educational context anywhere overrides
            if _EDUCATIONAL_CONTEXT.search(query):
                return SAFE
            return SafetyVerdict(blocked=True, category=category, message=_MESSAGES[category])
    return SAFE
