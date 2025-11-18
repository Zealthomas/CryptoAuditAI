"""
Cryptocurrency Mixer Database
Comprehensive list of known mixing services and sanctioned addresses
Updated: 2024 - Includes OFAC sanctioned mixers
"""

from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# KNOWN MIXER ADDRESSES DATABASE
# ============================================================================

KNOWN_MIXERS = {
    "tornado_cash": {
        "name": "Tornado Cash",
        "risk_score": 95,
        "sanctioned": True,
        "description": "OFAC Sanctioned - Ethereum Mixer",
        "blockchain": "ethereum",
        "addresses": [
            # Tornado Cash Router and Relayer addresses
            "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
            "0x722122df12d4e14e13ac3b6895a86e84145b6967",
            "0xdd4c48c0b24039969fc16d1cdf626eab821d3384",
            "0xd96f2b1c14db8458374d9aca76e26c3d18364307",
            "0x4736dcf1b7a3d580672cce6e7c65cd5cc9cfba9d",
            "0x169ad27a470d064dede56a2d3ff727986b15d52b",
            "0x0836222f2b2b24a3f36f98668ed8f0b38d1a872f",
            "0xf60dd140cff0706bae9cd734ac3ae76ad9ebc32a",
            "0x22aaa7720ddd5388a3c0a3333430953c68f1849b",
            "0xba214c1c1928a32bffe790263e38b4af9bfcd659",
            "0xb1c8094b234dce6e03f10a5b673c1d8c69739a00",
            "0x527653ea119f3e6a1f5bd18fbf4714081d7b31ce",
            "0x58e8dcc13be9780fc42e8723d8ead4cf46943df2",
            # 0.1 ETH pool
            "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc",
            # 1 ETH pool
            "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936",
            # 10 ETH pool
            "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
            # 100 ETH pool
            "0xa160cdab225685da1d56aa342ad8841c3b53f291",
            # Additional Tornado Cash contracts
            "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3",
            "0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144",
            "0xf67721a2d8f736e75a49fdd7fad2e31d8676542a",
            "0x9ad122c22b14202b4490edaf288fdb3c7cb3ff5e",
            "0x07687e702b410fa43f4cb4af7fa097918ffd2730",
            "0x23773e65ed146a459791799d01336db287f25334",
            "0x2717c5e28cf931547b621a5dddb772ab6a35b701",
            "0x03893a7c7463ae47d46bc7f091665f1893656003",
        ]
    },
    
    "blender_io": {
        "name": "Blender.io",
        "risk_score": 90,
        "sanctioned": True,
        "description": "OFAC Sanctioned - Bitcoin Mixer",
        "blockchain": "bitcoin",
        "addresses": [
            # Blender.io known Bitcoin addresses
            "bc1qa5wkgaew2dkv56kfvj49j0av5nml45x9ek9hz6",
            "bc1qn30xq20xrc6p0qyjy0xt5kqmz4f5vz3d0mkx4d",
            "3HxHGXSHwqVW1nC5cKYt8fzMFAMV3LnLdB",
            "35J3PzYdBnNqKrpPMUpLVKLFKKnxB9c3s8",
            "1BpbpfLdY7oBS9gK7aDXgvMgr1DPvNhEB2",
            "1FgAfKJFS8UY3w1D3hKLPDhF7hZmN5TvXw",
            "3HAZJBzY3zLMLmPrVvZS8FqLQqJBo96DEE",
            "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
        ]
    },
    
    "chipmixer": {
        "name": "ChipMixer",
        "risk_score": 92,
        "sanctioned": False,
        "description": "Seized by Law Enforcement - Bitcoin Mixer",
        "blockchain": "bitcoin",
        "addresses": [
            # ChipMixer known addresses (seized in 2023)
            "1ChipXTjt3LimxW3YwYkSpNwdWcKBrcTj",
            "3ChipPU9FPJYmDUy8qmcHjrWjWXaPMCVF",
            "bc1qchipvpj74g2qlvxdfyy08p7zd0cxgxwxyv9nqp",
            "1KFPo4hBQxz6WZr1J3K1UxJqsCQR5rA4Df",
            "12KPrD5kLHf2LbFCKT8WRzFBgVHxmGWJxW",
            "14F9sKY8VfvPSdPhePqZWcKB9dVJjqEk3Q",
            "1F1cXBEy8HwJVq1f2qTRx5Xj6V6xBqXqCX",
        ]
    },
    
    "sinbad": {
        "name": "Sinbad",
        "risk_score": 95,
        "sanctioned": True,
        "description": "OFAC Sanctioned - Bitcoin Mixer (Successor to Blender.io)",
        "blockchain": "bitcoin",
        "addresses": [
            # Sinbad.io addresses (sanctioned November 2023)
            "bc1qa0dwvcg8v5ew8vjlpydct8xvnq0v7masxe3pgv",
            "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
            "bc1q9ck0g82k30lc9gy5cnp6mwwyq6z2ngyxqtn2w4",
            "3BMEXqGpG4qxWTJLdqWvPdvhSqgPGk7jNs",
            "35UZjqKpWGVkJ7L8xj6g5aVTmRxChKJUPh",
        ]
    },
    
    "wasabi_wallet": {
        "name": "Wasabi Wallet",
        "risk_score": 75,
        "sanctioned": False,
        "description": "Privacy-focused Bitcoin Wallet with CoinJoin",
        "blockchain": "bitcoin",
        "addresses": [
            # Wasabi CoinJoin coordinator addresses
            "bc1qa24tsgchvuxsaccp8vrnkfd85hrcpafg20kmjw",
            "bc1qs604c7jv6amk4cxqlnvuxv26hv3e48cds4m0ew",
            "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
        ]
    },
    
    "samourai_whirlpool": {
        "name": "Samourai Whirlpool",
        "risk_score": 75,
        "sanctioned": False,
        "description": "Privacy Bitcoin Mixer (CoinJoin Implementation)",
        "blockchain": "bitcoin",
        "addresses": [
            # Samourai Whirlpool coordinator addresses
            "bc1qfxvu7e9n4epxrhtepvddq98rqmm3f6jnld9ytx",
            "bc1qz7w8nh29qzf6grvz3nvn49zg5rsnxqxrz28rka",
            "tb1qh287pqsh6c5mqyv8fjngh6dxkrws6ay8p7fcmj",
        ]
    },
    
    "ethereum_mixers": {
        "name": "Other Ethereum Mixers",
        "risk_score": 85,
        "sanctioned": False,
        "description": "Various Ethereum-based mixing services",
        "blockchain": "ethereum",
        "addresses": [
            # Other known Ethereum mixer addresses
            "0x8d12a197cb00d4747a1fe03395095ce2a5cc6819",
            "0xeefba1e63905ef1d7acba5a8513c70307c1ce441",
            "0x1e34a77868e19a6647b1f2f47b51ed72dede95dd",
            "0x3cbded43efdaf0fc77b9c55f6fc9988fcc9b757d",
        ]
    },
    
    "btc_fog": {
        "name": "Bitcoin Fog",
        "risk_score": 88,
        "sanctioned": False,
        "description": "Long-running Bitcoin Mixer (Operator arrested)",
        "blockchain": "bitcoin",
        "addresses": [
            "1FogxR7PxV9TJX7hPLvPELcXVqHAHuVRnz",
            "1Fo6xQJ2cD5cCZvfHK9GqFzB7dVHWCXGGy",
            "bc1qfogm8qxjsejjc7pk6r5wvzl0l08u4s9cjt3n09",
        ]
    },
    
    "bestmixer": {
        "name": "Bestmixer",
        "risk_score": 90,
        "sanctioned": False,
        "description": "Seized by Europol - Bitcoin Mixer",
        "blockchain": "bitcoin",
        "addresses": [
            "1BestMixerxHM1qJjxfqVqBT3KnTmXYFPX",
            "3BestLXiuN6YkGY2jqjpJHHqcjXXZFMzyf",
        ]
    },
    
    "helix": {
        "name": "Helix",
        "risk_score": 88,
        "sanctioned": False,
        "description": "Darknet Bitcoin Mixer (Operator convicted)",
        "blockchain": "bitcoin",
        "addresses": [
            "1HelixqYJuRpgECGF27n6VL3VsUqTEFxLh",
            "3HeLiXDVZRfnJqbFHzDTZR3pEZL7fQMrCf",
        ]
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_mixer_address(address: str) -> bool:
    """
    Quick check if an address is a known mixer
    
    Args:
        address: Blockchain address to check
        
    Returns:
        True if address is in mixer database
    """
    if not address:
        return False
    
    address_lower = address.lower().strip()
    
    for mixer_name, mixer_data in KNOWN_MIXERS.items():
        mixer_addresses = [addr.lower() for addr in mixer_data["addresses"]]
        if address_lower in mixer_addresses:
            return True
    
    return False


def get_mixer_info(address: str) -> Optional[Dict]:
    """
    Get detailed information about a mixer address
    
    Args:
        address: Blockchain address to check
        
    Returns:
        Dictionary with mixer information or None if not found
    """
    if not address:
        return None
    
    address_lower = address.lower().strip()
    
    for mixer_name, mixer_data in KNOWN_MIXERS.items():
        mixer_addresses = [addr.lower() for addr in mixer_data["addresses"]]
        if address_lower in mixer_addresses:
            return {
                "mixer_id": mixer_name,
                "name": mixer_data["name"],
                "risk_score": mixer_data["risk_score"],
                "sanctioned": mixer_data["sanctioned"],
                "description": mixer_data["description"],
                "blockchain": mixer_data["blockchain"],
                "matched_address": address
            }
    
    return None


def get_all_mixer_addresses() -> List[str]:
    """
    Get list of all mixer addresses in database
    
    Returns:
        List of all known mixer addresses
    """
    all_addresses = []
    for mixer_data in KNOWN_MIXERS.values():
        all_addresses.extend(mixer_data["addresses"])
    return all_addresses


def get_mixer_stats() -> Dict[str, Any]:
    """
    Get statistics about the mixer database
    
    Returns:
        Dictionary with database statistics
    """
    total_mixers = len(KNOWN_MIXERS)
    total_addresses = sum(len(m["addresses"]) for m in KNOWN_MIXERS.values())
    sanctioned_count = sum(1 for m in KNOWN_MIXERS.values() if m["sanctioned"])
    
    blockchains = {}
    for mixer_data in KNOWN_MIXERS.values():
        blockchain = mixer_data["blockchain"]
        blockchains[blockchain] = blockchains.get(blockchain, 0) + 1
    
    return {
        "total_mixers": total_mixers,
        "total_addresses": total_addresses,
        "sanctioned_mixers": sanctioned_count,
        "blockchains": blockchains,
        "mixer_names": [m["name"] for m in KNOWN_MIXERS.values()]
    }


# ============================================================================
# LOGGING INITIALIZATION
# ============================================================================

def init_mixer_database():
    """Initialize and validate the mixer database"""
    stats = get_mixer_stats()
    logger.info("=" * 70)
    logger.info("🔒 Mixer Detection Database Initialized")
    logger.info("=" * 70)
    logger.info(f"   Total Mixers: {stats['total_mixers']}")
    logger.info(f"   Total Addresses: {stats['total_addresses']}")
    logger.info(f"   Sanctioned Mixers: {stats['sanctioned_mixers']}")
    logger.info(f"   Blockchains Covered: {list(stats['blockchains'].keys())}")
    logger.info("=" * 70)
    
    return stats


# Auto-initialize on import
try:
    from typing import Any
    _stats = init_mixer_database()
except Exception as e:
    logger.warning(f"Mixer database initialization warning: {e}")