import torch
# torch.set_default_dtype(torch.float64)
from timeit import default_timer
from data_processing import UnitGaussianNormalizer, RangeNormalizer
from data_processing import CustomDataProcessorGraph
from utilities_src import MatReader, SquareMeshGenerator
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DataScaler = UnitGaussianNormalizer  # RangeNormalizer

TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

r = 4
s = int(((241 - 1)/r) + 1)
n = s**2
m = 100
k = 1

radius_train = 0.1
radius_test = 0.1

batch_size = 1
batch_size2 = 2

print('resolution', s)

ntrain = 100
ntest = 40

path = 'myscripts/GNOforPDEs/UAI1_r'+str(s)+'_n'+ str(ntrain)
path_model = 'model/'+path+''
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path+''
path_train_err = 'results/'+path+'train'
path_test_err16 = 'results/'+path+'test16'
path_test_err31 = 'results/'+path+'test31'
path_test_err61 = 'results/'+path+'test61'
path_image_train = 'image/'+path+'train'
path_image_test16 = 'image/'+path+'test16'
path_image_test31 = 'image/'+path+'test31'
path_image_test61 = 'image/'+path+'test61'

t1 = default_timer()
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u64 = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::4,::4].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::4,::4].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::4,::4].reshape(ntest,-1)

test_a = test_a.reshape(ntest,61,61)
test_a_smooth = test_a_smooth.reshape(ntest,61,61)
test_a_gradx = test_a_gradx.reshape(ntest,61,61)
test_a_grady = test_a_grady.reshape(ntest,61,61)
test_u = test_u.reshape(ntest,61,61)

test_a16 =test_a[:ntest,::4,::4].reshape(ntest,-1)
test_a_smooth16 = test_a_smooth[:ntest,::4,::4].reshape(ntest,-1)
test_a_gradx16 = test_a_gradx[:ntest,::4,::4].reshape(ntest,-1)
test_a_grady16 = test_a_grady[:ntest,::4,::4].reshape(ntest,-1)
test_u16 = test_u[:ntest,::4,::4].reshape(ntest,-1)
test_a31 =test_a[:ntest,::2,::2].reshape(ntest,-1)
test_a_smooth31 = test_a_smooth[:ntest,::2,::2].reshape(ntest,-1)
test_a_gradx31 = test_a_gradx[:ntest,::2,::2].reshape(ntest,-1)
test_a_grady31 = test_a_grady[:ntest,::2,::2].reshape(ntest,-1)
test_u31 = test_u[:ntest,::2,::2].reshape(ntest,-1)
test_a =test_a.reshape(ntest,-1)
test_a_smooth = test_a_smooth.reshape(ntest,-1)
test_a_gradx = test_a_gradx.reshape(ntest,-1)
test_a_grady = test_a_grady.reshape(ntest,-1)
test_u = test_u.reshape(ntest,-1)


meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
edge_index = meshgenerator.ball_connectivity(radius_train)
grid = meshgenerator.get_grid()

data_train = []
for j in range(ntrain):
    edge_attr = meshgenerator.attributes(theta=train_a[j,:])
    data_train.append(Data(x=torch.cat([grid, train_a[j,:].reshape(-1, 1),
                                        train_a_smooth[j,:].reshape(-1, 1), train_a_gradx[j,:].reshape(-1, 1), train_a_grady[j,:].reshape(-1, 1)
                                        ], dim=1),
                           y=train_u[j,:].view(-1, 1), coeff=train_a[j,:].view(-1, 1),
                           edge_index=edge_index, edge_attr=edge_attr,
                           ))

print('train grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[16,16])
edge_index = meshgenerator.ball_connectivity(radius_test)
grid = meshgenerator.get_grid()
data_test16 = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a16[j,:])
    data_test16.append(Data(x=torch.cat([grid, test_a16[j,:].reshape(-1, 1),
                                       test_a_smooth16[j,:].reshape(-1, 1), test_a_gradx16[j,:].reshape(-1, 1), test_a_grady16[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=test_u16[j, :].view(-1, 1), coeff=test_a16[j,:].view(-1, 1),
                           edge_index=edge_index, edge_attr=edge_attr,
                          ))

print('16 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[31,31])
edge_index = meshgenerator.ball_connectivity(radius_test)
grid = meshgenerator.get_grid()

data_test31 = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a31[j,:])
    data_test31.append(Data(x=torch.cat([grid, test_a31[j,:].reshape(-1, 1),
                                       test_a_smooth31[j,:].reshape(-1, 1), test_a_gradx31[j,:].reshape(-1, 1), test_a_grady31[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=test_u31[j, :].view(-1, 1), coeff=test_a31[j,:].view(-1, 1),
                           edge_index=edge_index, edge_attr=edge_attr,
                          ))

print('31 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[61,61])
edge_index = meshgenerator.ball_connectivity(radius_test)
grid = meshgenerator.get_grid()

data_test61 = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a[j,:])
    data_test61.append(Data(x=torch.cat([grid, test_a[j,:].reshape(-1, 1),
                                       test_a_smooth[j,:].reshape(-1, 1), test_a_gradx[j,:].reshape(-1, 1), test_a_grady[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=test_u[j, :].view(-1, 1), coeff=test_a[j,:].view(-1, 1),
                           edge_index=edge_index, edge_attr=edge_attr,
                          ))

print('61 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader16 = DataLoader(data_test16, batch_size=batch_size2, shuffle=False)
test_loader31 = DataLoader(data_test31, batch_size=batch_size2, shuffle=False)
test_loader61 = DataLoader(data_test61, batch_size=batch_size2, shuffle=False)


x = torch.cat([data.x for data in data_train], dim=0) 
y = torch.cat([data.y for data in data_train], dim=0)
edge_attr = torch.cat([data.edge_attr for data in data_train], dim=0)

input_encoder = DataScaler(dim=[0])
input_encoder.fit(x)

edge_attr_encoder = DataScaler(dim=[0])
edge_attr_encoder.fit(edge_attr)

output_encoder = DataScaler(dim=[0])
output_encoder.fit(y)
t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)


def load_dataset_graph():
    data_processor = CustomDataProcessorGraph(
        in_normalizer=input_encoder,
        edge_attr_normalizer=edge_attr_encoder,
        pos_normalizer=None,
        out_normalizer=output_encoder
        )
    test_loaders = {16: test_loader16, 31: test_loader31, 61: test_loader61}
    train_loaders = {61: train_loader}
    return train_loaders, test_loaders, data_processor

