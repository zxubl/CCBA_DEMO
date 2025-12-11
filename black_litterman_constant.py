weeks_per_year = 51

# prior_ratio =

# Asset list (representative indices)
assets = [
    'S&P 500 Index', 'SCHG', 'SCHV', 'IWM', 'MSCI Emerging Markets Index',
    'STOXX Europe 600 Index', 'MSCI China Index', 'TPX Index', 'MSCI Pacific ex Japan Index',
    'S&P Energy Sector Index', 'S&P Information Technology Sector Index',
    'Bloomberg Global Corporate Total Return Index', 'J.P. Morgan EMBI Global Diversified Index',
    'TLT', 'SHY', 'Bloomberg Global Inflation-Linked Bond Index', 'Bloomberg Global High Yield Total Return Index',
    'Bloomberg DXY Currency Index', 'MSCI Europe Currency Index', 'J.P. Morgan Emerging Markets Currency Index',
    'Bloomberg XAU Curncy', 'S&P GSCI Precious Metals Index',
    'MSCI Global Real Estate Index', 'Preqin Global Private Equity Index', 'HFRX Global Hedge Fund Index'
]

# Mapping from representative index to asset sub-class (资产子类)
sub_class_map = {
    'S&P 500 Index': '美股',
    'SCHG': '美股大型成长股',
    'SCHV': '美股大型价值股',
    'IWM': '美股小型股',
    'MSCI Emerging Markets Index': '新兴市场',
    'STOXX Europe 600 Index': '欧洲地区',
    'MSCI China Index': '亚太地区-中国',
    'TPX Index': '东京证交所指数',
    'MSCI Pacific ex Japan Index': '亚太地区（除日本）',
    'S&P Energy Sector Index': '行业指数-能源',
    'S&P Information Technology Sector Index': '行业指数-信息科技',
    'Bloomberg Global Corporate Total Return Index': '全球公司债',
    'J.P. Morgan EMBI Global Diversified Index': '新兴市场主权债',
    'TLT': '长端美国国债',
    'SHY': '短端美国国债',
    'Bloomberg Global Inflation-Linked Bond Index': '通胀挂钩债',
    'Bloomberg Global High Yield Total Return Index': '高收益公司债',
    'Bloomberg DXY Currency Index': '美元指数',
    'MSCI Europe Currency Index': '欧洲货币指数',
    'J.P. Morgan Emerging Markets Currency Index': '新兴市场货币',
    'Bloomberg XAU Curncy': '黄金',
    'S&P GSCI Precious Metals Index': '贵金属',
    'MSCI Global Real Estate Index': '房地产投资信托',
    'Preqin Global Private Equity Index': '私募股权',
    'HFRX Global Hedge Fund Index': '对冲基金'
}

# Reverse mapping from sub-class to index
reverse_sub_class_map = {v: k for k, v in sub_class_map.items()}

# Mapping from asset sub-class to broad asset class (资产大类)
sub_class_to_broad_class = {
    # 权益
    '美股': '权益',
    '美股大型成长股': '权益',
    '美股大型价值股': '权益',
    '美股小型股': '权益',
    '新兴市场': '权益',
    '欧洲地区': '权益',
    '亚太地区-中国': '权益',
    '东京证交所指数': '权益',
    '亚太地区（除日本）': '权益',
    '行业指数-能源': '权益',
    '行业指数-信息科技': '权益',

    # 债券
    '全球公司债': '债券',
    '新兴市场主权债': '债券',
    '长端美国国债': '债券',
    '短端美国国债': '债券',
    '通胀挂钩债': '债券',
    '高收益公司债': '债券',

    # 货币
    '美元指数': '货币',
    '欧洲货币指数': '货币',
    '新兴市场货币': '货币',

    # 大宗商品
    '黄金': '大宗商品',
    '贵金属': '大宗商品',

    # 另类投资
    '房地产投资信托': '另类投资',
    '私募股权': '另类投资',
    '对冲基金': '另类投资'
}

risk_to_target_vol = {
'C1': {
'Defensive': 0.056,
'Standard': 0.056,
'Enhanced': 0.057
},
'C2': {
'Defensive': 0.062,
'Standard': 0.065,
'Enhanced': 0.069
},
'C3': {
'Defensive': 0.097,
'Standard': 0.102,
'Enhanced': 0.107
},
'C4': {
'Defensive': 0.127,
'Standard': 0.133,
'Enhanced': 0.139
},
'C5': {
'Defensive': 0.16,
'Standard': 0.166,
'Enhanced': 0.172
}
}

