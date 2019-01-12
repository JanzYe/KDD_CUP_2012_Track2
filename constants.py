# -*- coding:utf-8 -*-

DIR_PATH = '/home/zhoudongliang/kddcup2012-track2/origin-data/track2/'
DATA_PATH = 'data/'
DATA_TRAINING = DIR_PATH + 'training.txt'
DATA_TEST = DIR_PATH + 'training.txt'
NUM_TRAINING = 149639105
NUM_TEST = 20297594

N_WORDS_QUERY = 1039180
N_WORDS_DESCRIPTION = 121996
N_WORDS_KEYWORD = 91482
N_WORDS_TITLE = 101182


PATH_USER = DIR_PATH + 'userid_profile.txt'
PATH_DESCRIPTION_ID = DIR_PATH + 'descriptionid_tokensid.txt'
PATH_KEYWORD_ID = DIR_PATH + 'purchasedkeywordid_tokensid.txt'
PATH_QUERY_ID = DIR_PATH + 'queryid_tokensid.txt'
PATH_TITLE_ID = DIR_PATH + 'titleid_tokensid.txt'
PATH_SOLUTION = DIR_PATH + 'KDD_Track2_solution.csv'

PATH_VEC_DESCRIPTION = DATA_PATH + 'descriptionid_tokensid.npz'
PATH_VEC_KEYWORD = DATA_PATH + 'purchasedkeywordid_tokensid.npz'
PATH_VEC_QUERY = DATA_PATH + 'queryid_tokensid.npz'
PATH_VEC_TITLE = DATA_PATH + 'titleid_tokensid.npz'
PATH_SUM_DESCRIPTION = DATA_PATH + 'descriptionid_tokensid.csv'
PATH_SUM_KEYWORD = DATA_PATH + 'purchasedkeywordid_tokensid.csv'
PATH_SUM_QUERY = DATA_PATH + 'queryid_tokensid.csv'
PATH_SUM_TITLE = DATA_PATH + 'titleid_tokensid.csv'
PATH_MUL_QUERY = 'data/multi_val_query.npz'
PATH_LEN_QUERY = 'data/valid_len_query.data'
PATH_MUL_TITLE = 'data/multi_val_title.npz'
PATH_LEN_TITLE = 'data/valid_len_title.data'
PATH_MUL_KEYWORD = 'data/multi_val_purchasedkeyword.npz'
PATH_LEN_KEYWORD = 'data/valid_len_purchasedkeyword.data'
PATH_MUL_DESCRIPTION = 'data/multi_val_description.npz'
PATH_LEN_DESCRIPTION = 'data/valid_len_description.data'


PATH_TRAIN = "data/feature_mapped_combined_train.data"
PATH_VALID = 'data/feature_mapped_combined_valid.data'
PATH_TEST = DIR_PATH + "combined_mapped_test.txt"

PATH_MIN_MAX = 'data/features_min_max.csv'

headers = [
    #               0           1         2          3             4        5           6            7
    'Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
    # 8            9             10        11       12          13
   'TitleID', 'DescriptionID', 'UserID', 'Gender', 'Age', 'RelativePosition',

    #  14            15                  16               17           18                 19               20
   'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',

    #  21            22             23              24              25
   'aCTR_Ad', 'aCTR_Advertiser', 'aCTR_Depth', 'aCTR_Position', 'aCTR_RPosition',

    # 26           27             28             29           30                 31               32
   'pCTR_Url', 'pCTR_Ad', 'pCTR_Advertiser', 'pCTR_Query', 'pCTR_Keyword', 'pCTR_Title', 'pCTR_Description',
    #  33            34            35               36
   'pCTR_User', 'pCTR_Gender', 'pCTR_Age', 'pCTR_RPosition',

    #  37            38                39
   'num_Depth', 'num_Position', 'num_RPosition',
    #  40            41              42            43
   'num_Query', 'num_Keyword', 'num_Title', 'num_Description',
    # 44                     45                  46               47               48
   'num_Imp__Ad', 'num_Imp__Advertiser', 'num_Imp_Depth', 'num_Imp_Position', 'num_Imp_RPosition',
   
    # 49                50                51             52               53       
    'sparse_Url', 'sparse_Ad', 'sparse_Advertiser', 'sparse_Depth', 'sparse_Position',
    # 54                55                   56             57                   58              59
    'sparse_Query', 'sparse_Keyword', 'sparse_Title', 'sparse_Description', 'sparse_UserID', 'sparse_Gender', 
    'sparse_Age', 'sparse_PosDepth'

   ]

feats = ['DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID', 'UserID',
         'Depth', 'Position', 'RelativePosition', 'Gender', 'Age']

CLICK = 'Click'
IMPRESSION = 'Impression'
SELF = 'Self'
TOKENS_LEN = 'TokensLen'
USER_ID = 'UserID'
AGE = 'Age'
GENDER = 'Gender'
QUERY_ID = 'QueryID'
KEYWORD_ID = 'KeywordID'
TITLE_ID = 'TitleID'
DESCRIPTION_ID = 'DescriptionID'
RELATIVE_POSITION = 'RelativePosition'
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

class FeatureStatistic:
    def __init__(self, name):
        self.name = name
        self.statistic = {}


class FeatureStat:
    def __init__(self, name):
        self.name = name
        self.stat = {CLICK: int(0), IMPRESSION: int(0), SELF: int(0), TOKENS_LEN: int(0)}

class FeatureNumerical:
    def __init__(self,name):
        self.name = name
        self.statistic = {}