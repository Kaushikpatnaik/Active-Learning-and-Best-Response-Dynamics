from datasets import *


data_dir = './data'

class WebData(RealDataSet):
	
	def __init__(self, shuffle = True, repeat = False, scale_p = None):
		path = data_dir + '/webdata.dat'
		self.d = 17
		self.name = 'WebData'
		RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
	
	def __str__(self):
		return self.name

class BreastCancer(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/wdbc.dat'
        self.d = 31
        self.name = 'Breastcancer'
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
   
    def __str__(self):
        return self.name

class Transfusion(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/transfusion.dat'
        self.d = 5
        self.name = 'Transfusion'
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name

class Mammography(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/mammography.dat'
        self.d = 6
        self.name = 'Mammography'
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

class Iris(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/iris-%d-%d.dat' % (i, j)
        self.name = 'Iris%d%d' % (i, j)
        self.d = 5
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
           
    def __str__(self):
        return self.name


class Wine(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/wine-%d-%d.dat' % (i, j)
        self.name = 'Wine%d%d' % (i, j)
        self.d = 14
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Seeds(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/seeds-%d-%d.dat' % (i, j)
        self.name = 'Seeds%d%d' % (i, j)
        self.d = 8
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Yeast(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/yeast-%d-%d.dat' % (i, j)
        self.name = 'Yeast%d%d' % (i, j)
        self.d = 9
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Mnist(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/Mnist_%d_%d' % (i, j)
        self.name = 'Mnist%d%d' % (i, j)
        self.d = 785
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name

class Fertility(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/fertility.dat'
        self.name = 'Fertility'
        self.d = 10
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name

class Spambase(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/spambase.dat'
        self.name = 'Spambase'
        self.d = 58
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
       
    def __str__(self):
        return self.name

class Ionosphere(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/ionosphere.dat'
        self.name = 'Ionosphere'
        self.d = 35
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name


class Biodeg(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/biodeg.dat'
        self.d = 42
        self.name = 'Biodeg'
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
    
    def __str__(self):
        return self.name


class CNAE(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/cnae-%d-%d.dat' % (i, j)
        self.name = 'CNAE%d%d' % (i, j)
        self.d = 857
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Semeion(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/semeion-%d-%d.dat' % (i, j)
        self.name = 'Semeion%d%d' % (i, j)
        self.d = 257
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Digits(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/pendigits-%d-%d.dat' % (i, j)
        self.name = 'Digits%d%d' % (i, j)
        self.d = 17
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Isolet(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/isolet-%d-%d.dat' % (i, j)
        self.name = 'Isolet%d%d' % (i, j)
        self.d = 618
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Letters(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/letters-%d-%d.dat' % (i, j)
        self.name = 'Letters%d%d' % (i, j)
        self.d = 17
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class Prover(RealDataSet):
    
    def __init__(self, i, j, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/prover-%d-%d.dat' % (i, j)
        self.name = 'Prover%d%d' % (i, j)
        self.d = 54
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
        
    def __str__(self):
        return self.name


class SPECTF(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/spectf.dat'
        self.name = 'SEPCTF'
        self.d = 45
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

class Particle(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/particle.dat'
        self.name = 'Particle'
        self.d = 51
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)
   
    def __str__(self):
        return self.name


class MAGIC(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/MAGIC.dat'
        self.name = 'MAGIC'
        self.d = 11
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

class DBWorld(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/dbworld.dat'
        self.name = 'DBWorld'
        self.d = 4703
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name


class ReutersGoldCoffee(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/reuters-gold-coffee.dat'
        self.name = 'ReutersGoldCoffee'
        self.d = 4972
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name


class ReutersNostopGoldCoffee(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/reuters-nostop-gold-coffee.dat'
        self.name = 'ReutersNoStopGoldCoffee'
        self.d = 4573
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

class ReutersShipSugar(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/reuters-ship-sugar.dat'
        self.name = 'ReutersShipSugar'
        self.d = 6991
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

class ReutersNostopShipSugar(RealDataSet):
    
    def __init__(self, shuffle = True, repeat = False, scale_p = None):
        path = data_dir + '/reuters-nostop-ship-sugar.dat'
        self.name = 'ReutersNostopShipSugar'
        self.d = 6563
        RealDataSet.__init__(self, path, shuffle, repeat, scale_p)

    def __str__(self):
        return self.name

# Dataset collections

# Low-dimensional
all_iris = [lambda: Iris(1, 2), lambda: Iris(1, 3), lambda: Iris(2, 3)]

all_seeds = [lambda: Seeds(1, 2), lambda: Seeds(1, 3), lambda: Seeds(2, 3)]

all_wine = [lambda: Wine(1, 2), lambda: Wine(1, 3), lambda: Wine(2, 3)]

all_yeast = [lambda i=i, j=j: Yeast(i, j) for i in range(1, 5) for j in range(i + 1, 5)]

all_mnist = [lambda i=i, j=j: Mnist(i, j) for i in range(0, 9) for j in range(i + 1, 9)]

all_digits = [lambda i=i, j=j: Digits(i, j) for i in range(10) for j in range(i + 1, 10)]

some_letters = [lambda j=j: Letters(0, j) for j in range(1, 26)]

low_d_singles = [Transfusion, Mammography, Fertility, MAGIC]

low_d = all_iris + all_seeds + all_wine + all_yeast + all_digits + some_letters + low_d_singles


# Mid-dimensional
all_prover = [lambda i=i, j=j: Prover(i, j) for i in range(6) for j in range(i + 1, 6)]

mid_d_singles = [BreastCancer, Ionosphere, SPECTF, Biodeg, Spambase, Particle]

mid_d = all_prover + mid_d_singles

# High-dimensional
some_isolet = [lambda j=j: Isolet(1, j) for j in range(2, 27)]

all_semeion = [lambda i=i, j=j: Semeion(i, j) for i in range(10) for j in range(i + 1, 10)]

all_cnae = [lambda i=i, j=j: CNAE(i, j) for i in range(1, 10) for j in range(i + 1, 10)]

some_reuters = [ReutersGoldCoffee,
                ReutersNostopGoldCoffee,
                ReutersShipSugar,
                ReutersNostopShipSugar]

high_d = all_semeion + all_cnae + some_isolet + some_reuters + [DBWorld]
          

all_real = low_d + mid_d + high_d


# Helper functions for testing datasets

def info(dataset):
    dataset.initialize()
    
    print dataset
    
    for i in range(5):
        try:
            x, y = dataset.next()
            print
            print 'x_%d:' % i, x
            print 'y_%d:' % i, y
        except StopIteration:
            break
    
    print
    try:
        instances = len(dataset.data)
        print 'instances: ', instances
    except:
        pass
    print 'dimensions:', x.shape[0]


def test_real_dataset(class_name, min_index = None, max_index = None):
    
    if min_index is None:
        datasets = [class_name]
    else:
        datasets = []
        for i in range(min_index, max_index + 1):
            for j in range(i + 1, max_index + 1):
                datasets.append(lambda i=i, j=j: class_name(i, j))
    
    for name in datasets:
        print
        dataset = name()
        print dataset
        dataset.initialize()
        print 'instances: ', len(dataset.data)
        d = dataset.data[0][0].shape[0]
        print 'raw dimensions:', d
        
        print 'checking...'
        while True:
            try:
                x, y = dataset.next()
            except StopIteration:
                break
            
            assert x.shape[0] == d + 1, x.shape[0]
            assert norm(x, 2) > 0.0, (x, norm(x, 2))
            assert y == -1 or y == 1, y
            
        print 'ok'
    
    print








