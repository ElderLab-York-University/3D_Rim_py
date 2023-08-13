
"""
Auto-generated download script for Thingi10K dataset.
Assuming the following python packages and external commands are available.

* argparse: for parse command line args.
* requests: for http communication.
* wget: for downloading files.

Usage:

    python Thingi10K_download.py

or

    python Thingi10K_download.py -o output_dir

"""

import argparse
import os.path
import sys
import requests
import os
import requests
import concurrent.futures
def download_file(file_id, output_dir):
    if not os.path.isdir(output_dir):
        raise IOError("Directory {} does not exist.".format(output_dir))

    url = "https://www.thingiverse.com/download:{}".format(file_id)
    r = requests.head(url)
    link = r.headers.get("Location", None)

    if link is None:
        print("File {} is no longer available on Thingiverse.".format(file_id))
        return

    # 使用函数获取干净的文件名
    output_file = os.path.join(output_dir, get_clean_filename_from_url(link))

    print("Downloading {}".format(output_file))

    response = requests.get(link, stream=True)
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("{} downloaded successfully.".format(output_file))

def get_clean_filename_from_url(url):
    # 去掉查询字符串
    filename = url.split('?')[0]
    # 从URL中获取文件名部分
    return filename.split('/')[-1]

def main():
    file_ids = [32770,34785,35269,36069,36082,36086,36090,36372,36373,37012,37093,37179,37266,37272,37274,37275,37276,37278,37280,37282,37283,37284,37285,37287,37288,37289,37322,37323,37384,37506,37620,37622,37627,37743,37744,37745,37750,37841,37865,37866,37880,37881,37883,37886,37888,37928,37964,37967,37968,37972,37991,38261,38290,38291,38292,38293,38294,38295,38296,38297,38464,38497,38498,38562,38636,38637,38638,38639,38640,38641,38643,38644,38645,38741,39010,39011,39012,39025,39026,39028,39050,39108,39109,39158,39159,39160,39164,39165,39166,39180,39182,39208,39245,39295,39344,39345,39349,39353,39355,39358,39495,39496,39498,39499,39505,39572,39573,39579,39636,39637,39638,39639,39641,39642,39643,39647,39648,39677,39678,39729,39730,39768,39769,39869,39893,39922,39923,39924,39927,39929,39930,39950,40051,40052,40067,40117,40118,40119,40171,40172,40179,40311,40312,40356,40359,40360,40443,40447,40448,40449,40450,40550,40583,40584,40585,40601,40602,40614,40660,40663,40666,40746,40836,40841,40842,40843,40844,40845,40846,40847,40848,40876,40877,40900,40915,40982,40984,40985,40986,40987,40988,40989,40990,40992,40993,40994,41098,41104,41105,41106,41107,41154,41178,41339,41360,41458,41466,41509,41521,41732,41929,41970,41984,41985,41997,42025,42035,42036,42040,42042,42155,42156,42192,42193,42194,42268,42370,42432,42468,42500,42501,42545,42575,42576,42606,42634,42638,42652,42721,42727,42822,42823,42836,42839,42844,42882,42936,42963,42974,42986,43096,43188,43189,43208,43296,43385,43386,43387,43388,43389,43390,43391,43392,43393,43394,43397,43399,43437,43439,43551,43554,43555,43584,43650,43651,43655,43663,43664,43665,43689,43690,43748,43780,43849,43850,43851,43852,43853,43858,43860,43868,43932,43959,43960,43988,43994,43995,44011,44016,44025,44057,44058,44060,44062,44063,44064,44100,44102,44110,44379,44380,44381,44382,44395,44396,44397,44398,44399,44400,44494,44498,44504,44633,44710,44740,44804,44899,44900,44901,44902,44903,44910,44921,44948,44949,44950,44951,44952,44998,45007,45009,45090,45169,45333,45370,45404,45408,45409,45410,45512,45513,45514,45515,45517,45550,45562,45611,45612,45613,45617,45620,45813,45814,46260,46261,46262,46265,46474,46519,46522,46527,46579,46665,46780,46840,47076,47095,47097,47098,47687,47732,47733,47748,47774,47775,47845,47851,47854,48031,48132,48223,48224,48270,48336,48338,48339,48340,48346,48367,48368,48415,48416,48417,48418,48419,48420,48421,48479,48495,48551,48552,48557,48558,48559,48940,49089,49105,49160,49163,49166,49167,49208,49374,49375,49376,49377,49378,49379,49380,49382,49383,49384,49423,49424,49529,49544,49545,49546,49547,49548,49549,49848,49852,49860,50112,50113,50114,50283,50428,50430,50450,50649,50650,50656,50659,50704,50725,50739,50741,50879,50881,51009,51010,51011,51012,51013,51014,51015,51016,51139,51490,51491,51492,51494,51508,51509,51510,51511,51637,51638,51639,51640,51641,51642,51643,51647,51648,51649,51654,51700,51797,51806,51807,51808,51809,51884,51910,52070,52071,52072,52073,52134,52135,52136,52137,52138,52139,52140,52141,52147,52410,52411,52563,52564,52733,52857,52858,52860,52862,53048,53404,53405,53406,53427,53429,53430,53431,53432,53433,53434,53435,53438,53439,53520,53581,53748,53749,53750,53754,53757,53835,53836,53858,53882,53884,53885,53886,54161,54162,54212,54215,54467,54468,54469,54470,54471,54725,54759,54820,55039,55040,55052,55092,55093,55094,55095,55177,55262,55277,55352,55353,55354,55440,55448,55506,55559,55577,55580,55583,55660,55914,55922,55963,56029,56093,56094,56096,56099,56101,56109,56265,56497,56498,56524,56592,56630,56954,56955,56956,57106,57140,57191,57330,57331,57332,57355,57357,57419,57420,57421,57422,57467,57500,57587,57656,57657,57658,57659,57713,57714,57800,57809,57810,57811,57812,57827,57854,57908,57937,57942,57993,58007,58008,58009,58010,58011,58012,58020,58021,58069,58070,58072,58238,58261,58393,58439,58564,58874,58876,58877,58878,58879,58933,58938,59079,59125,59126,59158,59197,59226,59228,59229,59230,59301,59333,59440,59492,59494,59497,59546,59559,59560,59703,59704,59705,59706,59707,59709,59710,59747,59752,59753,59756,59757,59758,59760,59767,59769,59770,59936,59942,60094,60099,60100,60101,60246,60262,60265,60266,60270,60273,60275,60276,60277,60514,60515,60516,60517,60550,60554,60568,60584,60716,60848,60861,60862,60863,60866,60880,60916,60918,60927,61039,61083,61085,61136,61192,61199,61242,61418,61431,61458,61462,61464,61583,61585,61669,61755,61767,61792,62286,62287,62288,62415,62570,62571,62572,62592,62593,62860,62880,62987,62990,62991,62992,62998,63234,63244,63245,63422,63446,63447,63448,63451,63452,63769,63785,63871,64577,64578,64579,64821,64957,64983,64992,65002,65003,65004,65143,65144,65149,65150,65281,65282,65395,65400,65402,65414,65443,65444,65504,65561,65562,65563,65564,65582,65583,65584,65585,65586,65587,65588,65603,65604,65607,65608,65609,65610,65612,65614,65615,65616,65617,65618,65619,65641,65642,65643,65644,65754,65755,65756,65757,65758,65778,65779,65904,65942,66108,66478,66479,66480,66481,66482,66484,66485,66486,66566,66783,66902,67120,67121,67223,67331,67408,67497,67501,67516,67523,67524,67550,67551,67792,67817,67821,67856,67857,67919,67923,67924,67926,67929,67985,67986,67987,67988,67989,67990,67991,67992,67994,68199,68203,68215,68248,68255,68334,68335,68370,68371,68372,68380,68382,68416,68500,68509,68510,68511,68645,68646,68647,68648,68649,68651,68653,68658,68659,68661,68742,68812,68813,68814,68934,68935,69056,69057,69058,69079,69085,69088,69089,69090,69091,69092,69109,69128,69260,69264,69265,69267,69325,69399,69402,69403,69404,69405,69406,69537,69548,69685,69689,69693,69694,69696,69697,69701,69702,69729,69730,69880,69883,69896,69899,69902,69910,69915,69917,69928,69930,69931,69932,69940,69951,69965,69978,69983,69984,69985,69986,70043,70044,70045,70046,70058,70060,70066,70369,70370,70371,70373,70376,70377,70379,70380,70381,70561,70829,70830,70831,70832,70833,70834,70835,70836,70837,70838,70839,70840,70921,70922,70923,71041,71046,71260,71261,71263,71264,71265,71266,71277,71307,71316,71328,71329,71330,71331,71382,71383,71461,71462,71709,71711,71712,71722,71760,71762,71763,71764,71919,71920,71987,71990,71995,71999,72001,72093,72100,72101,72107,72189,72190,72191,72203,72214,72225,72227,72337,72419,72581,72582,72603,72607,72613,72658,72665,72666,72668,72669,72760,72881,72947,72960,73082,73085,73086,73088,73089,73090,73135,73152,73153,73154,73155,73156,73157,73158,73160,73161,73162,73163,73164,73183,73265,73409,73412,73413,73414,73416,73417,73427,73428,73430,73431,73464,73474,73486,73700,73736,73737,73738,73739,73866,73868,73986,74126,74151,74154,74155,74157,74439,74447,74449,74455,74457,74458,74461,74492,74494,74497,74642,74690,74730,74890,74932,74933,74934,74944,75090,75098,75101,75106,75113,75114,75115,75116,75117,75118,75119,75121,75133,75142,75232,75269,75282,75283,75359,75361,75622,75623,75652,75653,75976,75984,75986,75989,76051,76052,76053,76054,76055,76056,76057,76143,76192,76198,76199,76200,76202,76204,76205,76206,76207,76291,76391,76475,76477,76538,76572,76573,76579,76581,76679,76680,76681,76682,76683,76684,76685,76710,76712,76714,76723,76724,76755,76756,76778,76901,76903,76907,76908,76947,77011,77013,77014,77015,77016,77017,77019,77020,77185,77335,77336,77340,77342,77343,77461,77463,77472,77474,77477,77508,77844,77912,77916,77917,77918,77919,77920,77921,77922,77926,77927,77928,77929,77930,77931,77932,77933,77934,77935,77936,77938,77939,77940,77941,77944,77945,77947,77948,77949,77951,78211,78251,78253,78295,78298,78319,78322,78323,78324,78325,78327,78328,78329,78351,78481,78733,78736,78738,78894,78895,78896,78944,78954,78976,79073,79177,79182,79183,79184,79185,79189,79191,79193,79194,79195,79197,79200,79202,79241,79279,79729,79741,79810,79851,79861,79936,79938,79939,79940,79954,79955,79956,79961,80084,80086,80353,80354,80356,80357,80361,80363,80364,80365,80366,80370,80414,80557,80603,80604,80657,80746,80748,80750,80752,80755,80758,80760,80769,80937,80938,80939,80940,80941,80942,80945,81090,81148,81152,81181,81221,81223,81259,81260,81261,81263,81264,81265,81290,81291,81307,81308,81309,81310,81366,81636,81637,81776,81878,81919,81936,82058,82059,82170,82183,82257,82377,82378,82379,82380,82392,82394,82407,82471,82526,82536,82537,82539,82663,82664,82667,82668,83002,83005,83006,83166,83167,83255,83256,83416,83490,83491,83649,83650,83651,83716,83720,83835,84022,84023,84024,84030,84032,84033,84042,84319,84320,84321,84615,84618,84621,84623,84639,84640,84641,84642,84736,84737,84757,84758,84781,84914,84915,84916,84917,84918,84930,84931,84942,84943,84975,84977,84987,85112,85152,85379,85430,85440,85493,85495,85550,85580,85659,85770,85774,85826,85860,85881,85886,86164,86235,86236,86238,86239,86240,86241,86242,86593,87355,87391,87392,87393,87395,87398,87429,87430,87431,87432,87519,87520,87599,87602,87659,87660,87687,87688,87689,87690,87721,87825,88053,88147,88566,88567,88577,88578,88579,88580,88581,88614,88615,88616,88626,88628,88855,88991,89266,89415,89416,89417,89418,89420,89421,89422,89448,89449,89451,89517,89518,89519,89527,89528,89535,89538,89540,89541,89542,89634,89855,89895,89911,89912,89913,89914,90088,90089,90207,90223,90224,90226,90273,90275,90277,90279,90280,90281,90282,90283,90286,90287,90358,90359,90429,90430,90431,90432,90433,90434,90435,90437,90441,90442,90443,90687,90688,90980,91006,91013,91023,91024,91025,91142,91145,91146,91147,91206,91223,91224,91390,91392,91408,91452,91453,91454,91455,91474,91606,91607,91610,91611,91655,91656,91657,91658,91904,91915,91917,91918,92116,92118,92119,92152,92155,92415,92420,92421,92422,92614,92668,92820,92832,92834,92836,92837,92944,92952,92964,93067,93070,93071,93072,93073,93107,93108,93109,93370,93540,93543,93545,93547,93549,93551,93553,93554,93556,93702,93703,93936,94016,94018,94019,94058,94059,94139,94140,94143,94144,94146,94147,94148,94149,94150,94152,94153,94155,94156,94157,94158,94203,94215,94216,94477,94482,94490,94492,94662,94664,94665,94668,94674,94675,94676,94677,94731,94732,94733,94734,94888,94990,95002,95037,95180,95185,95315,95316,95476,95486,95487,95488,95489,95494,95496,95497,95498,95500,95502,95503,95663,95775,95776,95779,95782,95794,95795,95796,95797,95798,95799,95800,95801,95807,95808,95809,95810,95811,95812,95813,96046,96047,96065,96067,96068,96069,96070,96071,96074,96076,96077,96130,96132,96160,96455,96456,96457,96458,96583,96653,96654,96656,96657,96658,96659,96660,96661,96662,96752,96755,96770,96771,96772,96775,96776,96777,96778,96779,96780,96781,96914,96915,96957,96963,96966,96967,97124,97126,97128,97130,97156,97161,97165,97166,97168,97170,97172,97174,97175,97177,97179,97254,97473,97474,97478,97488,97489,97490,97493,97494,97503,97507,97511,97512,97514,97590,97593,97596,97657,97658,97659,97660,97661,97674,97686,97729,97730,97733,97734,97859,97912,97939,97940,97941,97942,98006,98009,98012,98017,98019,98021,98163,98165,98330,98350,98351,98412,98413,98478,98479,98480,98535,98567,98625,98651,98659,98660,98663,98763,98765,98767,98797,98938,98939,98940,98976,99265,99267,99269,99270,99272,99275,99276,99467,99468,99469,99776,99812,99813,99836,99864,99865,99866,99898,99900,99904,99905,99906,99907,99908,99933,99992,100028,100034,100070,100077,100322,100323,100336,100339,100340,100342,100343,100344,100345,100349,100388,100423,100478,100506,100507,100638,100639,100640,100642,100643,100645,100646,100815,100831,100887,100894,101170,101187,101330,101333,101337,101339,101340,101341,101342,101548,101550,101556,101558,101560,101619,101620,101634,101635,101636,101864,101953,102041,102066,102248,102249,102250,102255,102294,102295,102296,102625,103086,103141,103142,103143,103144,103145,103146,103194,103196,103284,103286,103289,103355,103356,103357,103358,103538,103826,104301,104400,104401,104402,104403,104404,104431,104442,104445,104512,104559,104563,104582,104639,104640,104737,104738,104929,104930,104968,104969,105103,105120,105171,105220,105226,105230,105330,105336,105338,105413,105414,105637,105685,105686,105687,105688,105690,105691,105692,105695,105696,105761,105762,105765,105803,105859,105860,105862,105924,106039,106090,106100,106146,106368,106557,106558,106559,106681,106683,106705,106725,106777,106778,106779,106781,106782,106783,106784,106785,106786,106787,106788,106789,106884,107007,107041,107046,107066,107067,107068,107069,107070,107071,107148,107149,107295,107389,107390,107391,107392,107402,107434,107438,107439,107440,107445,107558,107586,107587,107910,108147,108199,108293,108328,108329,108330,108337,108341,108517,108518,108521,108613,108616,108618,108619,108620,108693,108771,108785,108858,108884,109014,109015,109130,109131,109136,109138,109176,109344,109375,109459,109572,109576,109729,109772,109925,110028,110042,110044,110172,110173,110201,110202,110203,110204,110205,110207,110211,110213,110353,110357,110358,110375,110444,110445,110699,110711,110715,110782,110786,110793,110795,110796,110801,110803,110834,110837,110905,110906,110907,110908,110909,110910,110911,110950,111006,111008,111009,111010,111011,111013,111021,111046,111048,111240,111394,111400,111552,111599,112230,112539,112541,112544,112546,112916,112919,112920,112921,112922,112925,112926,112928,112931,112932,112940,112941,112965,112996,112997,113000,113001,113125,113126,113127,113128,113134,113221,113340,113341,113342,113343,113344,113418,113419,113420,113653,113855,113858,113861,113862,113863,113864,113865,113866,113867,113868,113869,113887,113889,113890,113891,113908,114020,115053,115601,116051,116063,116064,116065,116066,116067,116069,116070,116073,116074,116075,116076,116078,116291,116292,116293,116294,116863,116870,116871,116872,116875,116877,116878,116880,116890,116891,116893,116896,117150,117423,117428,117523,117525,117526,117647,117952,118032,118034,118035,118037,118287,118288,118290,118292,118293,118295,118297,118298,118299,118301,118302,118304,118306,118307,118309,118317,118675,118676,118677,118678,118679,118680,118681,118682,118683,118768,118882,118893,118895,118904,118905,118909,118914,118916,118920,118921,119238,119247,119829,119830,119918,120755,120756,120787,120788,120789,120790,120791,120792,121118,121122,121394,121395,121396,121397,121398,121399,121400,121401,121404,121868,121869,122308,122309,122349,122351,122390,122482,122550,122551,122866,122868,122870,123043,123044,123045,123787,123958,124034,124035,124036,124038,124039,124040,124041,124043,124186,124249,124336,124358,124360,124362,124363,124367,124368,124369,124370,124372,124373,124374,124375,124532,124533,124534,124535,124682,124707,125066,125067,125068,125069,125071,125947,126195,126196,126197,126282,126283,126284,126286,126544,126660,126813,126820,127151,127243,127891,128057,128065,128076,128350,128373,128482,128483,128484,128485,128558,128559,128640,128915,129288,129867,129871,129872,129889,129890,129891,129892,129894,129895,129896,129898,129899,129906,129907,129908,129909,129910,129911,129912,129914,129915,129916,129917,129918,129919,129920,129922,129924,129925,129926,129927,129928,129929,129930,129931,129932,129936,129937,129973,129988,129992,130014,130015,130017,130030,130031,130046,130788,130960,130961,130962,130964,130967,130969,130972,130973,130975,130977,130978,130979,130980,130986,131090,131091,131453,131601,131602,131604,131605,131720,131723,131725,131727,131730,131969,132057,132060,132420,132421,132422,132423,132425,132426,132428,132429,132430,132431,132433,132434,132435,132436,132438,132440,132441,132442,132444,132446,132447,132448,132449,132450,132536,132560,133078,133079,133275,133276,133991,133992,133993,134543,134601,134618,134619,134621,134622,134630,134631,134633,135059,135062,135064,135065,135066,135071,135074,135077,135078,135214,135215,135216,135217,135218,135219,135220,135221,135222,135363,135364,135365,135493,135671,135672,135674,135675,135676,135677,135790,135803,135805,135806,135807,135808,135810,136023,136074,136235,136236,136243,136404,136523,136634,136635,136636,136638,136639,136640,136935,137707,137976,137977,138557,138581,139430,139764,139765,139767,139934,140511,140534,140545,140547,144106,144422,145328,145329,145330,145331,148292,148293,151360,151365,151370,151376,152775,153100,153368,153852,153943,153948,153951,153956,153959,153960,153961,153963,153964,153968,157174,158248,158249,158250,159693,159694,159695,159696,161091,161092,161093,161094,161461,161717,161718,162100,162282,162283,162331,162332,162333,162336,162337,162338,162575,162861,162862,162863,162890,162892,162893,163021,163758,163760,163761,163762,163763,166279,167064,167107,167108,167109,167110,167113,168074,168075,168076,168077,168078,168079,168908,168928,169526,170134,174325,175915,177007,177008,177009,178192,178193,178194,178195,178196,178197,178199,178200,178201,178202,178226,185290,186199,186200,186201,186202,186203,186204,186541,186544,186545,186546,186548,186551,186552,186553,186554,186559,187279,187280,187281,187282,187283,188402,188403,188495,188496,188497,188498,188499,188500,188501,188502,188503,188505,188506,189404,190247,190248,190249,190253,190399,190401,190402,190415,191077,191078,191079,191081,191968,192042,194884,195680,195696,195697,196121,196122,196123,196126,196127,196187,196188,196189,196190,196191,196192,196193,196194,196196,196208,196209,199814,199818,199820,200003,200004,200005,200006,200009,200010,200011,200074,200075,200076,200077,200078,200079,200080,200081,200082,200083,200212,200369,200683,200684,200685,200687,200688,200689,200690,200691,200692,200693,200694,200695,200696,200697,200961,200962,200963,200964,200965,200966,200967,200968,200969,200973,201958,201959,201960,201962,202265,203217,203218,203280,203281,203282,203289,203344,204267,204477,204951,204952,204953,204954,204955,204956,204957,204958,204959,204960,204961,204962,205450,206494,206495,208367,208547,209086,211915,212094,212098,212133,212134,212135,212136,212138,213384,213386,213387,213388,213389,213390,214246,214251,214255,214479,214480,214482,215845,215846,215990,215991,215992,216762,216763,217015,217016,217030,217031,218730,218731,220653,221686,222140,224026,224504,224829,225937,225939,225940,225941,225945,225946,225947,225949,225950,225951,225952,225953,225954,225955,225956,225957,225958,225959,225960,225961,225962,225963,225965,225966,225967,225968,225969,225970,225971,225972,225973,225977,225978,226160,226633,226954,227387,227388,227615,227616,228070,228302,229596,229598,229599,229600,229601,229602,229603,229604,229605,229606,229607,230152,230153,230349,230909,232909,232912,232916,233186,233187,233188,233189,233190,233191,233192,233193,233194,233195,233196,233197,233198,233199,235279,235725,235726,236142,236173,237623,237624,237626,237630,237632,237633,237634,237635,237640,237643,237735,237736,237737,237739,238413,238414,238419,238420,238422,238618,238635,238636,238637,238638,238639,239180,239181,239182,239183,239188,239191,240292,240293,240294,241235,242305,243015,243016,243018,247069,247402,247674,248185,248186,248188,248189,248190,248191,248192,248193,249518,249519,249520,249521,249522,249523,249524,250042,250393,250394,250395,250396,250397,250398,250400,250402,250739,251163,251402,252119,252614,252631,252632,252634,252635,252636,252640,254558,255172,255495,255497,255522,255657,255658,257178,257585,257586,257589,257593,257594,258879,260537,263198,263199,263200,263225,263989,264707,265592,265728,265730,267657,267659,267661,267663,267666,267784,269117,269118,269119,269120,269121,269122,269123,269124,269125,269126,269127,269128,269129,269800,269801,270392,270393,270456,270940,271300,271301,271302,271303,271304,271305,271550,271864,271865,271866,271868,271869,271870,271871,271872,271873,271874,271875,271876,271877,274379,276932,276933,276935,276936,276937,276938,278455,278456,278458,280278,280280,280281,280282,280283,280285,280654,283342,285606,285607,285610,286158,286160,286161,286986,286987,287448,288349,288350,288351,288352,288353,288354,288355,289646,289648,289649,289650,289651,289652,289653,289654,289655,289658,289659,289660,289661,289662,289663,289664,289665,289666,289667,289668,289669,289670,289671,289672,289673,289674,289675,289676,289677,289678,289679,289681,289682,289683,289685,289686,289687,289688,289689,289690,289691,289692,290506,293137,295746,295749,296801,296802,296803,296804,296805,299291,301065,301066,301798,301804,301915,301920,303901,303902,311321,311322,311327,311328,311329,311331,313444,313873,313875,313877,313878,313879,313880,313881,313882,313883,313884,313917,314000,314002,314003,314437,314438,314439,314440,314441,314442,314586,314587,314588,314590,314591,314592,314969,316358,318043,318044,319831,326895,331045,331105,332810,333704,338509,343529,343530,343531,343533,343534,343535,343536,343537,343538,343541,343542,343543,343544,343546,343547,343548,343549,343550,343551,343552,343553,343554,343555,343556,347932,347938,350273,350274,350275,350276,350277,351479,353686,354371,354668,358257,360006,360067,360068,360069,360070,360072,360073,360074,362398,363825,364253,365170,365370,366248,366249,366250,366253,366254,366255,366279,366361,366727,366956,366957,367343,368147,368149,368889,368891,369988,370518,370884,370885,370886,370887,372055,372056,372602,374397,374398,375065,375081,375237,375239,375240,375241,375242,375244,375245,375247,375248,375249,375250,375251,375252,375253,375254,375255,375257,375258,375260,375261,375263,375264,375265,375266,375267,375268,375269,375270,375271,375273,375274,375275,375276,375277,375278,375279,375280,375281,376235,376243,376257,376259,383021,383022,383023,384086,384087,384373,384574,384575,387222,387871,387872,388482,389250,389251,390916,390936,395055,398259,398304,399559,399560,399561,399562,399563,399564,399566,399567,399899,400235,400236,400237,400238,402963,405822,405883,407639,409624,409628,409629,409633,409634,409635,409637,416558,416683,416684,422367,422368,422369,422370,423818,423825,425600,425601,428602,428603,428604,428605,428606,428607,434512,441707,441708,441709,441710,441711,441713,441714,441715,441716,441717,441718,441719,441720,441722,441723,441725,441726,441727,441728,442382,442383,442384,442385,442386,442387,442388,442389,442391,447331,447333,447334,449817,449818,449998,449999,454346,461109,461110,461111,461112,461115,462509,462510,462511,462512,462513,462514,462515,462516,462517,462518,462519,462520,462521,462522,462523,462524,462525,462526,462527,462528,462529,462530,462531,462533,462534,462535,462537,462538,462539,462540,462541,462542,462543,462544,462545,462546,462547,462548,462549,464842,464843,464846,466263,470186,470464,471983,471984,471985,471986,471987,471988,471989,471990,471992,471993,471994,471995,471996,471998,471999,472000,472001,472002,472004,472008,472009,472011,472015,472016,472017,472018,472019,472020,472021,472022,472023,472025,472026,472027,472028,472029,472030,472031,472032,472033,472034,472035,472036,472037,472038,472039,472040,472041,472042,472043,472045,472046,472047,472048,472049,472050,472051,472052,472053,472054,472055,472056,472057,472058,472059,472060,472061,472062,472063,472064,472065,472067,472068,472069,472070,472071,472072,472073,472074,472075,472076,472077,472078,472079,472080,472081,472082,472083,472084,472085,472086,472087,472088,472089,472092,472093,472094,472095,472096,472097,472098,472099,472100,472101,472102,472103,472105,472106,472107,472108,472111,472112,472113,472114,472115,472116,472117,472118,472120,472121,472122,472123,472124,472125,472126,472129,472130,472131,472132,472133,472134,472135,472136,472137,472138,472139,472140,472141,472142,472143,472144,472145,472146,472147,472148,472149,472150,472152,472153,472154,472155,472156,472157,472158,472159,472160,472161,472162,472163,472164,472165,472166,472167,472168,472169,472171,472172,472173,472174,472175,472176,472178,472179,472180,472181,472182,472183,472185,472186,472187,472189,472190,472191,472192,472193,472194,472196,472197,472199,472200,472201,472202,472203,472204,472205,472206,472820,474822,474823,474824,474825,479928,479929,479930,479931,482738,482910,482916,486859,486860,488046,488047,488048,488049,488050,488051,490704,491062,494998,495000,496387,496389,496669,496670,496671,496672,496673,497408,497513,497978,498430,498431,498432,498713,498714,500086,500087,500088,500089,500090,500091,500092,500093,500094,500095,500096,500097,500098,500099,500100,500101,500102,500103,500104,500105,500106,500107,500108,500109,500110,500111,500112,500113,500114,500115,500116,500117,500118,500119,500120,501146,501148,501149,501151,501152,501154,503412,509308,509309,509310,509311,509312,509313,509314,509315,509316,509317,509318,509319,509320,511803,514852,514853,518029,518030,518031,518032,518033,518034,518035,518036,518037,518038,518039,518079,518080,518081,518082,518083,518084,518085,518086,518087,518088,518089,518090,518091,518092,518093,518094,518095,519507,520644,520645,520646,520647,520648,521303,521419,521599,522979,527582,531010,533453,534903,535201,535202,535629,535691,543706,543708,548859,548860,548861,548862,548863,548864,548865,548867,548868,548871,548872,548875,550198,550807,550964,551019,551020,551021,551074,551075,557384,560534,564455,565652,565653,567026,567027,567028,567030,567031,567032,567033,567034,567035,567036,567037,567038,567039,567040,567041,567042,568153,569947,570154,574283,577765,579369,579370,579371,591211,591212,591213,591214,591215,591216,591217,591218,591219,591221,591223,591224,591225,591227,591228,591229,591230,591231,591232,591233,591234,591235,591238,591239,591242,591243,591244,591245,591246,591247,591248,591249,591250,593082,593083,593084,593381,593382,593383,593385,593387,595269,601644,601645,601646,601647,611675,611676,613009,619534,619537,619538,619539,619567,619569,619671,621685,621686,622000,623170,623171,623172,627212,627213,627441,630796,630797,635746,635747,635748,635749,635750,635751,635752,635753,635754,636794,636795,636796,636797,636798,636799,636801,636802,636803,636804,636805,636806,636807,636808,636809,636810,636811,636812,636813,636814,636816,636817,636818,636819,639929,639930,641138,641139,641140,641141,641142,641143,641144,641145,641146,642501,642502,649193,650178,650179,650180,650181,650182,650183,650184,650185,650187,656071,662071,663109,667995,668056,668057,668418,668419,669962,669963,669964,669966,669967,669969,669970,669971,669978,669979,673086,673783,674844,674845,674846,674847,674848,677339,677340,677341,677342,682288,682291,682292,686056,687430,688366,688367,693468,693469,693471,693473,697192,697201,697202,697204,697205,697206,697207,697208,697209,697210,697211,697212,697213,697215,697216,697217,697218,697219,697220,697222,697223,697225,697226,697227,697228,697612,700903,700904,700905,700906,700907,700908,701637,702203,702204,702384,702385,702386,702387,702388,702389,702390,702392,702395,702396,702398,702400,702401,702402,702403,702404,702405,702406,702407,702411,702412,702414,702417,702418,702419,702420,702421,702422,702423,702424,702426,703556,703557,703558,703800,709758,711199,718339,719791,721775,723890,723891,723892,723893,723895,723900,723901,726664,726665,734278,734279,734280,734960,734961,734962,734963,735907,736023,736024,736124,740391,740392,740393,741525,741528,741529,741532,747838,754638,756802,756803,756805,756871,756872,756873,760497,760498,762584,762586,762592,762593,762595,762596,762599,762600,762601,762602,762603,762604,762605,762606,762975,763398,763714,763718,771609,777037,777038,777039,777040,777041,779996,780017,780018,780047,780460,787646,788566,789069,789070,789071,789072,789073,789074,789075,789076,789077,789078,789080,789081,789082,789083,789084,789085,789086,789088,789796,793564,793565,796150,799433,799434,799435,799437,799439,804293,804294,804297,804298,805914,814657,814659,814661,814665,814666,814667,814670,814673,814675,814677,814678,814679,814757,814761,815474,815475,815476,815477,815478,815480,815481,815483,815484,815485,815486,816583,816584,816585,821369,822696,822697,822698,827639,827640,827641,827772,827773,834595,834600,834601,837999,838000,838001,838003,839690,839694,839722,839723,839724,840364,841122,842507,842651,844209,848343,849720,849721,849722,849726,849727,849728,849729,852088,856325,856326,856327,856328,856330,856331,856334,858851,858852,858853,858854,858856,860050,894828,894829,894830,894831,894832,894833,894834,894835,894836,904472,904473,904475,904476,904477,904478,904479,904480,904505,904506,904509,905406,905407,906183,906891,908459,908460,914490,914672,914673,914674,914675,914676,914677,914678,914679,914680,914681,914682,914683,914684,914686,916993,917938,919984,919985,919986,919987,919991,919992,919993,919994,919995,921796,925244,929561,931874,931886,931887,932610,932611,932612,934258,940381,940382,940414,956446,956447,956669,958469,958471,958472,958473,958474,958475,958476,958477,958478,958479,958480,960066,964933,972829,972830,972835,973980,977256,985159,985160,985161,985343,985716,991313,996797,996809,996810,996813,1005285,1016856,1016874,1018273,1018274,1018275,1018276,1018278,1018279,1018295,1020669,1021166,1028706,1036309,1036310,1036311,1036312,1036313,1036315,1036316,1036318,1036319,1036392,1036393,1036394,1036395,1036396,1036398,1036399,1036400,1036402,1036428,1036429,1036462,1036464,1036466,1036467,1036468,1036473,1036650,1036651,1036652,1036653,1036654,1036655,1036657,1036658,1043461,1043471,1044251,1044252,1047582,1047980,1050061,1051177,1053374,1053375,1053377,1053568,1053874,1053875,1054518,1063851,1063852,1063853,1063854,1063855,1063857,1063858,1063860,1063861,1063862,1063863,1063866,1065030,1066896,1066897,1066898,1066899,1066900,1066901,1069352,1071710,1075456,1075457,1075458,1079828,1080515,1080516,1082149,1082231,1083348,1083349,1083350,1083351,1083352,1083353,1083354,1085685,1085686,1087135,1088224,1088281,1093600,1095206,1095207,1095208,1095209,1095210,1095211,1097847,1097848,1099667,1103938,1120760,1120761,1120762,1120763,1120764,1120765,1120766,1120767,1120768,1120770,1120771,1120773,1120774,1120775,1120777,1120778,1120779,1120780,1124376,1129073,1129074,1129076,1129078,1129079,1129080,1129081,1129082,1129083,1129086,1130077,1130078,1130082,1146162,1146164,1146165,1146166,1146168,1146169,1146172,1146175,1146178,1146180,1146181,1146183,1146184,1146185,1146186,1146187,1146188,1146189,1146190,1146192,1146193,1146195,1146196,1146199,1146260,1146261,1147240,1148096,1148449,1150177,1156747,1158261,1158262,1158263,1158264,1158265,1158266,1158267,1158268,1158269,1164424,1172870,1179042,1179043,1185140,1203191,1207650,1207651,1207655,1207658,1207659,1207662,1207663,1207664,1207665,1207666,1207667,1207670,1213332,1215417,1223672,1223673,1223674,1223675,1224383,1224384,1224385,1224386,1224387,1228189,1228190,1228191,1228192,1228193,1228194,1228195,1228198,1228200,1231075,1233192,1233194,1233195,1255206,1258096,1273629,1273630,1275114,1275115,1275116,1275117,1277085,1282328,1288648,1288649,1288651,1288652,1291011,1307396,1307397,1307398,1307399,1307400,1312949,1312950,1312951,1312952,1312953,1312954,1312955,1312957,1312958,1312959,1312960,1312962,1312968,1312969,1312970,1312971,1313533,1313544,1313801,1315830,1315831,1315832,1315833,1315834,1315835,1315837,1315838,1315839,1315840,1315841,1315842,1315844,1315845,1320334,1322464,1322465,1322469,1324574,1329184,1329185,1329188,1335002,1336194,1341427,1341742,1341744,1341745,1344037,1344038,1344039,1344040,1344041,1344043,1344044,1344047,1344048,1344051,1344052,1344053,1344055,1344057,1349593,1351747,1356634,1356637,1356640,1356644,1356652,1368076,1378611,1382593,1382595,1382596,1382597,1382598,1382600,1382601,1384321,1384322,1387346,1395582,1395584,1395585,1396885,1396886,1396888,1396890,1396891,1396892,1396893,1396897,1396898,1396899,1396900,1396901,1396906,1397588,1399249,1411169,1411686,1411687,1411696,1411699,1411700,1411702,1417953,1417954,1417955,1417956,1417958,1417959,1417960,1417962,1417963,1417965,1417966,1432740,1432741,1435439,1435440,1436278,1439534,1439537,1451897,1451898,1451899,1451900,1451901,1452668,1452669,1452672,1452674,1452676,1452677,1452678,1452679,1453582,1454014,1454015,1454016,1454017,1454018,1454387,1454388,1454389,1455630,1455661,1458669,1458670,1458671,1458674,1458682,1458684,1458685,1458687,1458688,1458689,1458691,1458692,1458693,1458694,1458695,1458698,1458699,1458700,1458701,1458702,1458703,1460441,1464288,1472695,1472696,1481523,1481537,1482988,1482989,1482990,1483721,1484067,1489589,1489590,1489591,1489592,1489593,1489594,1489595,1489596,1489597,1489598,1489599,1489600,1489676,1490842,1490843,1490844,1492345,1494102,1501544,1501545,1501546,1501547,1501548,1501549,1501590,1501591,1502911,1504272,1505017,1505018,1505019,1505020,1505022,1505023,1505025,1505026,1505029,1505031,1505032,1505036,1512169,1514493,1514494,1514496,1514498,1514499,1514502,1514900,1514901,1514903,1514904,1517909,1517923,1519467,1519468,1519469,1519471,1533026,1533028,1535134,1535136,1535137,1535138,1535139,1536353,1536354,1536355,1536356,1559370,1559483,1566089,1566090,1582374,1582376,1582377,1582378,1582379,1582380,1582381,1582382,1582383,1582384,1582385,1582386,1582387,1582388,1582389,1582390,1582391,1582392,1582393,1582394,1582395,1582396,1582397,1582398,1582399,1582400,1582401,1582402,1582403,1582404,1582405,1582406,1582407,1582408,1582409,1582410,1582411,1582412,1582413,1582414,1582415,1582416,1582417,1582419,1582420,1582421,1582422,1582423,1582424,1582425,1582426,1582427,1582428,1582429,1582430,1582431,1582432,1582433,1582434,1582435,1582436,1582437,1582438,1582439,1592665,1592666,1600246,1600247,1601406,1614064,1620054,1620055,1620057,1620061,1620063,1620064,1620066,1621241,1623064,1624039,1626828,1628448,1628449,1628454,1628455,1646897,1646931,1652975,1652976,1681115,1681117,1681118,1681119,1681120,1681122,1681123,1681124,1681182,1681183,1681184,1706456,1706457,1706459,1706461,1706462,1706463,1706465,1706466,1706472,1706475,1706476,1716284,1716285,1717684,1717685,1722410,1722411,1743321,1743322,1772543,1772591,1772593,1777452];
    output_dir = "../Models/"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_file, file_id, output_dir) for file_id in file_ids]

        for future in concurrent.futures.as_completed(futures):
            # 如果有任何异常，这里会抛出
            future.result()

if __name__ == "__main__":
    main();
