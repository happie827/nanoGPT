"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

char_codec = 'utf-8'
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding=char_codec) as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")


# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")


# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 2,045,389
# all the unique characters: 
#  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_abcdefghijklmnopqrstuvwxyz|~ ²·× ​–―‘’“”…‧′※ℓⅡ↑→↓∙∼①②③⑥⑦■▲△▶▷◆◇○●☞✅　〈〉《》「」『』くㄱㄴㄷㄹㅁㅇㅋㆍ㈜㎏㎚㎜㎝㎞㎟㎡㎢㎥上下东中主乙京人代件任份企低住佛保信修假價兆先兒內全兵典凡出切別前副加動化北半华協南印反受古史合同名命和問善器國土地型場外太失奧姚子孫學定家富實審射對導小尹山岸崔崛州巨己巳市幅年度庸張強强從循德心必志忠恒患愚愛憂憐戰手技抗援收攻政故敵文新斷日最月期本朴李株案梁棟樂檢正步武死毒氣水汎決沈治法洞浮润減滴濟無狂獨率班現球環生産用田甲界異病發的益盟目相省知石破社神私稅種空立端範經線罰美群義考者而聞職股脫腐與芳英茂草蒙蔡號血行街裝見視親言記訪評說談謀謝議護責賞賢質起超路車軍软追通達郞都釋野鋒錦長限雄難雲電露靜非韓風馬高鳥鸿麻齒가각간갇갈갉감갑값갓갔강갖같갚갛개객갤갭갯갱걀거걱건걷걸검겁것겉게겐겔겠겨격겪견결겸겹겼경곁계곗고곡곤곧골곪곰곱곳공곶과곽관괄광괘괜괴굉교굣구국군굳굴굵굶굼굽굿궁궂궈권궐궜궤귀귄규균귤그극근글긁금급긋긍기긱긴길김깁깃깅깊까깎깐깔깜깝깡깥깨깬깰깼꺼꺾껀껄껍껏껐껑께껴꼈꼬꼭꼴꼼꼽꽁꽂꽃꽉꽤꾀꾸꾹꾼꿀꿈꿎꿔꿨꿰뀌뀐뀔끄끈끊끌끓끔끕끗끝끼낀낄낌낏나낙난날낡남납낫났낭낮낯낳내낸낼냄냅냈냉냐냥너넉넋넌널넓넘넝넣네넥넨넬넷넸녀녁년념녔녕노녹논놀놈놉농높놓놔놨뇌뇨누눅눈눌눔눕눠눴뉘뉜뉴늄느는늘늠능늦늪늬니닉닌닐님닙닛닝다닥닦단닫달닭닮닳담답닷당닿대댄댈댐댑댓댔댕더덕던덜덟덤덧덩덫덮데덱덴델뎀뎌뎠도독돈돋돌돔돕동돼됐되된될됨됩두둑둔둘둠둡둥둬뒀뒤뒷듀듈드득든듣들듬듭듯등디딕딘딛딜딤딥딩딪따딱딴딸땀땄땅때땐땠땡떠떡떤떨떫떴떻떼뗀또똑똘똥똬뚜뚝뚫뚱뛰뛴뛸뜀뜨뜩뜬뜯뜰뜸뜻띄띈띠띤띨라락란랄람랍랏랐랑랗래랙랜랠램랩랫랬랭랴략량러럭런럴럼럽럿렀렁렇레렉렌렘렛렜려력련렬렴렵렷렸령례롄로록론롤롬롭롯롱뢰료룡루룩룬룰룸룹룻룽뤄뤘뤼류륙륜률륨륭르른를름릅릇릉리릭린릴림립릿링마막만많말맑맘맙맛망맞맡매맥맨맴맵맷맸맹맺머먹먼멀멈멋멍메멕멘멜멤멩며면멸몄명몇모목몫몬몰몸몹못몽뫼묘무묵묶문묻물뭄뭇뭉뭍뭐뭔뭘뮤뮬므미믹민믿밀밋밌밍및밑바박밖반받발밝밟밤밥방밭배백밴밸뱀뱅버벅번벌범법벗벙베벡벤벨벳벼벽변별볍병볕보복볶본볼봄봅봇봉봐봤봬부북분불붉붐붓붕붙뷔뷰브븐블비빅빈빌빔빗빙빚빛빠빡빤빨빴빵빼뺀뺄뺏뺐뺨뻐뻑뻔뻗뻥뼈뽀뽐뽑뾰뿌뿐뿔뿜쁘쁜쁠쁨삐사삭산살삶삼삽삿샀상샅새색샌샐샘생샤샬샴샵샷샹섀서석섞선섣설섬섭섯섰성세섹센셀셈셉셋셌셍셔션셜셧셨셰셴셸소속솎손솔솜솟송솥쇄쇠쇳쇼숄숍숏수숙순숟술숨숫숭숱숲쉐쉘쉬쉰쉴쉼쉽슈슐슘슝스슨슬슭슴습슷승싀시식신싣실싫심십싱싶싸싹싼쌀쌈쌌쌍쌓써썩썬썰썸썹썼썽쎄쏘쏙쏜쏟쏠쏴쐈쐐쐬쑤쑥쑹쓰쓱쓴쓸씀씁씌씨씩씬씰씹씻씽아악안앉않알앓암압앗았앙앞애액앤앨앰앱앳앴앵야약얀얄얇얏양얕얘어억언얹얻얼얽엄업없엇었엉엌엎에엑엔엘엠엣엥여역엮연열염엽엿였영옅옆예옛오옥온올옭옮옳옴옵옷옹와왁완왈왑왓왔왕왜외왼요욕욘용우욱운울움웃웅워웍원월웠웨웬웰웹위윅윈윌윔윗윙유육윤율융으은을음읍응의이익인일읽잃임입잇있잉잊잎자작잔잖잘잠잡잣잤장잦재잭잰잼잿쟁저적전절젊점접젓정젖제젝젠젤젬젯져졌조족존졸좀좁종좇좋좌좡죄죈죠주죽준줄줌줍중줘줬쥐쥔즈즉즌즐즘증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짰짱째쨌쩌쩍쩐쩔쪼쪽쫌쫑쫓쭈쭉쭐쭤쯔쯤쯩찌찍찐찔찜찝찢차착찬찮찰참찼창찾채책챈챌챔챗챘챙처척천철첨첩첫청체첸첼쳉쳐쳤초촉촌촘촛총촨촬촹최쵝추축춘출춤춥춧충춰췄췌취츄츠측츰층치칙친칠침칩칫칭카칵칸칼캄캐캔캘캠캡캣커컨컫컬컴컵컸케켄켈켐켓켜켤켰코콕콘콜콤콥콧콩콰콴쾌쿄쿠쿡쿤쿨쿼퀀퀄퀘퀴퀵퀸퀼큐큘크큰클큼큽키킥킨킬킴킵킷킹타탁탄탈탐탑탓탔탕태택탠탤탬탭탰탱터턱턴털텀텁텃텅테텍텐텔템텡텨텼톈토톡톤톨톰톱통퇴투툰툴툼퉈튀튄튈튕튜튠튬트특튼튿틀틈틍틔티틱틴틸팀팁팅파팍팎판팔팜팝팟팠팡패팩팬팰팸팹팻팽퍼펀펄펌펑페펙펜펠펩펫펭펴편펼폄폈평폐포폭폰폴폼폿퐁표푸푹푼풀품풋풍퓨퓰프픈플픔피픽핀필핌핍핏핑하학한할함합핫항해핵핸햄햇했행햐향허헌험헛헝헤헥헨헬헴혀혁현혈혐협혔형혜호혹혼홀홈홋홍화확환활황회획횟횡효후훈훌훔훗훙훤훨훼휘휜휠휩휴흉흐흑흔흘흙흠흡흥흩희흰히힌힐힘힙不率利李％，．＼ｇｍ｜📌
# vocab size: 1,846
# train has 1,840,850 tokens
# val has 204,539 tokens



