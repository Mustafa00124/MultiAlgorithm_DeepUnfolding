import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--threshold_all', action='store_true')
parser.add_argument('--file_write', type=str, default='C:/Users/Mustafa/Downloads/MultiAlgorithm_DeepUnfolding/Results.csv')
parser.add_argument('--layer_list', type=lambda s: [item for item in s.split(',')], default=['Median2_Layer', 'Mean2_Layer'], help='List of layer names separated by comma')
parser.add_argument('--writecsv', action='store_true', help='Flag to write results to CSV')
parser.add_argument('--name', type=str, default='Madu')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--initial-lr', type=float, default=0.003)
parser.add_argument(
    "--lr_decay_rate", type=float, default=0.3
)  # learning rate decay rate
parser.add_argument(
    "--lr_decay_intv", type=int, default=30
)  # learning rate decay interval
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seed", default=123)
parser.add_argument("--time-steps", type=int, default=50)
parser.add_argument("--patch-size", default=None)
parser.add_argument("--max-size", type=int, default=128)
parser.add_argument("--crop-size", default=None)
parser.add_argument('--model', default='roman_r')
parser.add_argument("--coeff-L", type=float, default=.1)
parser.add_argument("--coeff-S", type=float, default=0.05)
parser.add_argument("--coeff-Sside", type=float, default=.001)
parser.add_argument("--reweighted", default=False, action='store_true')
parser.add_argument("--l1-l2", default=False, action='store_true')
parser.add_argument('--l1l1', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--log-dir', default= None)
parser.add_argument('--load', default=None, type=str, help='Load and test given checkpoint')
parser.add_argument('--loss_type', default='L_tversky_bce')
parser.add_argument('--clamp-coeff-L', default=.05)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--data_path', type=str, default=r'C:\Users\Mustafa\Downloads\dataset2014\dataset')

parser.add_argument('-hf', '--hidden-filters', type=int, default=8)

parser.add_argument('--category', type=int, default=39)
parser.add_argument('--split', type=float, default=0)

parser.add_argument('--mrpca-rho', type=float, default=None)
parser.add_argument('--mrpca-gamma', type=float, default=None)

cfgs = parser.parse_args()

cfgs.categories = ['baseline/highway', 'baseline/office', 'baseline/pedestrians', 'baseline/PETS2006']
cfgs.include_in_train = [(), (), (), ()]
if cfgs.category == 0:
    cfgs.categories = ['baseline/pedestrians']
    cfgs.include_in_train = [(300, 450, 640, 940)]
elif cfgs.category == 1:
    cfgs.categories = ['baseline/PETS2006']
    cfgs.include_in_train = [(740, 980, 370)]
elif cfgs.category == 2:
    cfgs.categories = ['baseline/highway']
    cfgs.include_in_train = [[]]
elif cfgs.category == 3:
    cfgs.categories = ['baseline/office']
    cfgs.include_in_train = [(710, 1370, 1710)]
elif cfgs.category == 4:
    cfgs.categories = ['lowFramerate/port_0_17fps']
    cfgs.include_in_train = [(1070, 1160, 1850, 1900)]
elif cfgs.category == 5:
    cfgs.categories = ['lowFramerate/tramCrossroad_1fps']
    cfgs.include_in_train = [(400, 500, 600)]
elif cfgs.category == 6:
    cfgs.categories = ['lowFramerate/tunnelExit_0_35fps']
    cfgs.include_in_train = [(2010, 2700)]
elif cfgs.category == 7:
    cfgs.categories = ['lowFramerate/turnpike_0_5fps']
    cfgs.include_in_train = [(800, 900, 1000)]
elif cfgs.category == 8:
    cfgs.categories = ['thermal/corridor']
    cfgs.include_in_train = [[]]
elif cfgs.category == 9:
    cfgs.categories = ['thermal/diningRoom']
    cfgs.include_in_train = [(1220, 2120)]
elif cfgs.category == 10:
    cfgs.categories = ['thermal/lakeSide']
    # cfgs.include_in_train = [(2000, 2660, 3250, 5830)]
    cfgs.include_in_train = [(1750, 2500, 3450, 6000)]
elif cfgs.category == 11:
    cfgs.categories = ['thermal/library']
    cfgs.include_in_train = [[]]
elif cfgs.category == 12:
    cfgs.categories = ['thermal/park']
    cfgs.include_in_train = [(290, 500)]
elif cfgs.category == 13:
    cfgs.categories = ['badWeather/blizzard']
    cfgs.include_in_train = [(1000, 1250, 1540, 3360)]
elif cfgs.category == 14:
    cfgs.categories = ['badWeather/skating']
    cfgs.include_in_train = [(1360, 1640)]
elif cfgs.category == 15:
    cfgs.categories = ['badWeather/snowFall']
    cfgs.include_in_train = [(810, 2780)]
elif cfgs.category == 16:
    cfgs.categories = ['badWeather/wetSnow']
    # cfgs.include_in_train = [(510, 600)]
    cfgs.include_in_train = [(550, 1700)]
elif cfgs.category == 17:
    cfgs.categories = ['dynamicBackground/boats']
    cfgs.include_in_train = [(1950, 7100, 7400, 7900)]
elif cfgs.category == 18:
    cfgs.categories = ['dynamicBackground/canoe']
    cfgs.include_in_train = [[940]]
elif cfgs.category == 19:
    cfgs.categories = ['dynamicBackground/fall']
    cfgs.include_in_train = [(2050, 2590)]
elif cfgs.category == 20:
    cfgs.categories = ['dynamicBackground/fountain01']
    cfgs.include_in_train = [(720, 1150)]
elif cfgs.category == 21:
    cfgs.categories = ['dynamicBackground/fountain02']
    cfgs.include_in_train = [[680]]
elif cfgs.category == 22:
    cfgs.categories = ['dynamicBackground/overpass']
    cfgs.include_in_train = [(2420, 2630)]
elif cfgs.category == 23:
    cfgs.categories = ['shadow/backdoor']
    cfgs.include_in_train = [(1370, 1480, 1750, 1900)]
elif cfgs.category == 24:
    cfgs.categories = ['shadow/bungalows']
    cfgs.include_in_train = [(700, 1430)]
elif cfgs.category == 25:
    cfgs.categories = ['shadow/busStation']
    cfgs.include_in_train = [(1000, 600)]
elif cfgs.category == 26:
    cfgs.categories = ['shadow/copyMachine']
    cfgs.include_in_train = [(1060, 2620)]
elif cfgs.category == 27:
    cfgs.categories = ['shadow/cubicle']
    cfgs.include_in_train = [(1620, 6160)]
elif cfgs.category == 28:
    cfgs.categories = ['shadow/peopleInShade']
    cfgs.include_in_train = [(300, 600, 1060)]
elif cfgs.category == 29:
    cfgs.categories = ['nightVideos/bridgeEntry']
    cfgs.include_in_train = [(1600, 1700)]
elif cfgs.category == 30:
    cfgs.categories = ['nightVideos/busyBoulvard']
    cfgs.include_in_train = [(800, 1580)]
elif cfgs.category == 31:
    cfgs.categories = ['nightVideos/fluidHighway']
    cfgs.include_in_train = [(420, 610)]
elif cfgs.category == 32:
    cfgs.categories = ['nightVideos/streetCornerAtNight']
    cfgs.include_in_train = [(970, 2760)]
elif cfgs.category == 33:
    cfgs.categories = ['nightVideos/tramStation']
    cfgs.include_in_train = [(990, 1460)]
elif cfgs.category == 34:
    cfgs.categories = ['nightVideos/winterStreet']
    cfgs.include_in_train = [(900, 1220)]
elif cfgs.category == 35:
    cfgs.categories = ['turbulence/turbulence0']
    cfgs.include_in_train = [(1800, 2170, 2350)]
elif cfgs.category == 36:
    cfgs.categories = ['turbulence/turbulence1']
    cfgs.include_in_train = [(1850, 2250)]
elif cfgs.category == 37:
    cfgs.categories = ['turbulence/turbulence2']
    cfgs.include_in_train = [(660, 2300)]
elif cfgs.category == 38:
    cfgs.categories = ['turbulence/turbulence3']
    cfgs.include_in_train = [(940, 1160)]
elif cfgs.category == 47:
    cfgs.categories = ['cameraJitter/badminton']
    cfgs.include_in_train = [[]]
elif cfgs.category == 48:
    cfgs.categories = ['cameraJitter/boulevard']
    cfgs.include_in_train = [(1200, 1960)]
elif cfgs.category == 49:
    cfgs.categories = ['cameraJitter/sidewalk']
    cfgs.include_in_train = [[820]]
elif cfgs.category == 50:
    cfgs.categories = ['cameraJitter/traffic']
    cfgs.include_in_train = [[]]

elif cfgs.category == 65:
    # Regular videos (traffic, ped...)
    cfgs.categories = ['baseline/PETS2006', 'shadow/backdoor', 'shadow/bungalows']
elif cfgs.category == 66:
    # Regular videos (traffic, ped...)
    cfgs.categories = ['baseline/highway', 'baseline/pedestrians', 'shadow/peopleInShade']

elif cfgs.category == 67:
    # Video with bad weather or dynamic background
    cfgs.categories = ['badWeather/skating', 'dynamicBackground/canoe', 'dynamicBackground/fall', 'turbulence/turbulence3']
elif cfgs.category == 68:
    # Video with bad weather or dynamic background
    cfgs.categories = ['badWeather/snowFall', 'dynamicBackground/fountain02', 'turbulence/turbulence2']

elif cfgs.category == 69:
    # Video with camera shake
    cfgs.categories = ['cameraJitter/badminton', 'cameraJitter/traffic']
elif cfgs.category == 70:
    # Video with camera shake
    cfgs.categories = ['cameraJitter/boulevard', 'cameraJitter/sidewalk']

elif cfgs.category == 71:
    cfgs.categories = ['baseline/PETS2006', 'shadow/backdoor', 'shadow/bungalows', 'badWeather/skating', 'dynamicBackground/canoe', 'dynamicBackground/fall', 'turbulence/turbulence3']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]
elif cfgs.category == 72:
    cfgs.categories = ['baseline/highway', 'baseline/pedestrians', 'shadow/peopleInShade', 'badWeather/snowFall', 'dynamicBackground/fountain02', 'turbulence/turbulence2']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]
    
elif cfgs.category == 73:
    cfgs.categories =  ['baseline/PETS2006', 'shadow/backdoor', 'shadow/bungalows', 'cameraJitter/badminton', 'cameraJitter/traffic']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]
elif cfgs.category == 74:
    cfgs.categories = ['baseline/highway', 'baseline/pedestrians', 'shadow/peopleInShade', 'cameraJitter/boulevard', 'cameraJitter/sidewalk']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]

elif cfgs.category == 75:
    cfgs.categories = ['baseline/PETS2006', 'shadow/backdoor', 'shadow/bungalows', 'badWeather/skating', 'dynamicBackground/canoe', 'dynamicBackground/fall', 'turbulence/turbulence3', 'cameraJitter/badminton', 'cameraJitter/traffic']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]
elif cfgs.category == 76:
    cfgs.categories = ['baseline/highway', 'baseline/pedestrians', 'shadow/peopleInShade', 'badWeather/snowFall', 'dynamicBackground/fountain02', 'turbulence/turbulence2', 'cameraJitter/boulevard', 'cameraJitter/sidewalk']
    cfgs.include_in_train = [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]



