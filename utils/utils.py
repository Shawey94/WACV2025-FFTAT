import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import networkx as nx
import copy
import cpnet
import random
from itertools import combinations, groupby
import matplotlib.pyplot as plt

def mix_img(x_s, x_t, x_s_mask, x_t_mask, patches_in_cores, influenceRatio):
    x_s_clone = x_s.clone()
    index = np.argsort(x_s_mask.ravel())[:-patches_in_cores-1:-1]
    pos = np.unravel_index(index, x_s_mask.shape)
    pos = np.column_stack(pos)
    x_s_swap_index = pos.tolist()

    x_t_clone = x_t.clone()
    index = np.argsort(x_t_mask.ravel())[:-patches_in_cores-1:-1]
    pos = np.unravel_index(index, x_t_mask.shape)
    pos = np.column_stack(pos)
    x_t_swap_index = pos.tolist()

    core_div = int(patches_in_cores / 16)
    core_mod = int(patches_in_cores % 16)
    
    replace1 = 0   #[0,15]
    replace2 = 0   #[0,15]
    for temp_pos in x_s_swap_index:
        temp_x = temp_pos[0]
        temp_y = temp_pos[1]
        if(temp_x < core_div or (temp_x == core_div and temp_y <= core_mod) ):
            continue
        while( [replace1,replace2] in x_s_swap_index):
            replace2 +=1 
            if(replace2 > 15):
                replace1 += 1
                replace2 = 0
            continue
        x_s_clone[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1] = x_s[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1]
        x_s_clone[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1] = x_s[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1]
        replace2 +=1 
        if(replace2 > 15):
            replace1 += 1
            replace2 = 0
    
    replace1 = 0   #[0,15]
    replace2 = 0   #[0,15]
    for temp_pos in x_t_swap_index:
        temp_x = temp_pos[0]
        temp_y = temp_pos[1]
        if(temp_x < core_div or (temp_x == core_div and temp_y <= core_mod) ):
            continue
        while( [replace1,replace2] in x_t_swap_index):
            replace2 +=1 
            if(replace2 > 15):
                replace1 += 1
                replace2 = 0
            continue
        x_t_clone[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1] = x_t[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1]
        x_t_clone[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1] = x_t[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1]
        replace2 +=1 
        if(replace2 > 15):
            replace1 += 1
            replace2 = 0
    '''
    plt.subplot(4,1,1)
    plt.imshow(np.asarray(x_s.cpu()).transpose(1,2,0))      
    plt.subplot(4,1,2)
    plt.imshow(np.asarray(x_s_clone.cpu()).transpose(1,2,0))
    plt.subplot(4,1,3)
    plt.imshow( np.asarray(x_t.cpu()).transpose(1,2,0) )
    plt.subplot(4,1,4)
    plt.imshow( np.asarray(x_t_clone.cpu()).transpose(1,2,0) )
    '''
    
    x_t_clone_clone = x_t_clone.clone()
    x_s_clone_clone = x_s_clone.clone()
    #add partial x_t to x_s
    x_s_clone_clone[:,int(core_div*16): int((core_div+1)*16), int(core_mod*16):256] = influenceRatio * x_t_clone[:,int(core_div*16):int(core_div+1)*16, int(core_mod*16):256] + x_s_clone[:,int(core_div*16):int(core_div+1)*16, int(core_mod*16):256] 
    x_s_clone_clone[:,int((core_div+1)*16):256, :] = influenceRatio * x_t_clone[:,int((core_div+1)*16):256, :] + x_s_clone[:,int((core_div+1)*16):256, :]

    #core
    x_s_clone_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] = x_s_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] + influenceRatio/2 * x_t_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] 
    x_s_clone_clone[:,0:int((core_div-1)*16), :] = x_s_clone[:,0:int((core_div-1)*16), :] + influenceRatio/2 * x_t_clone[:,0:int((core_div-1)*16), :]

    #add partial x_s to x_t
    x_t_clone_clone[:,int(core_div*16): int((core_div+1)*16), int(core_mod*16):256] = influenceRatio * x_s_clone[:,int(core_div*16):int(core_div+1)*16, int(core_mod*16):256] + x_t_clone[:,int(core_div*16):int(core_div+1)*16, int(core_mod*16):256] 
    x_t_clone_clone[:,int((core_div+1)*16):256, :] = influenceRatio * x_s_clone[:,int((core_div+1)*16):256, :] + x_t_clone[:,int((core_div+1)*16):256, :]

    #core
    x_t_clone_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] = x_t_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] + influenceRatio/2 * x_s_clone[:,int(core_div*16):int((core_div+1)*16), 0:int(core_mod*16)] 
    x_t_clone_clone[:,0:int((core_div-1)*16), :] = x_t_clone[:,0:int((core_div-1)*16), :] + influenceRatio/2 * x_s_clone[:,0:int((core_div-1)*16), :]

    '''
    plt.subplot(4,1,1)
    plt.imshow(np.asarray(x_s.cpu()).transpose(1,2,0))      
    plt.subplot(4,1,2)
    plt.imshow(np.asarray(x_s_clone.cpu()).transpose(1,2,0))
    plt.subplot(4,1,3)
    plt.imshow( np.asarray(x_t.cpu()).transpose(1,2,0) )
    plt.subplot(4,1,4)
    plt.imshow( np.asarray(x_t_clone.cpu()).transpose(1,2,0) )
    plt.show()        
    '''
    return x_s_clone_clone, x_t_clone_clone

def re_org_img(img, mask, patches_in_cores):
    img_clone = img.clone()

    index = np.argsort(mask.ravel())[:-patches_in_cores-1:-1]
    pos = np.unravel_index(index, mask.shape)
    pos = np.column_stack(pos)
    #pos_copy = copy.deepcopy(pos)     
    
    swap_index = pos.tolist()
    #pos_copy = pos_copy.tolist()
    core_div = patches_in_cores / 16
    core_mod = patches_in_cores % 16
    
    '''
    img_clone_temp = img_clone.clone()
    for i in swap_index:
        div = i[0]
        mod = i[1]
        img_clone_temp[:,(div * 16):((div+1)* 16):1, (mod* 16):((mod+1)* 16):1] = torch.zeros(3, 16,16)
    plt.imshow(np.asarray(img_clone_temp.cpu()).transpose(1,2,0))
    plt.show() 
    '''
    replace1 = 0   #[0,15]
    replace2 = 0   #[0,15]
    for temp_pos in swap_index:
        temp_x = temp_pos[0]
        temp_y = temp_pos[1]
        if(temp_x < core_div or (temp_x == core_div and temp_y <= core_mod) ):
            continue
        while( [replace1,replace2] in swap_index):
            replace2 +=1 
            if(replace2 > 15):
                replace1 += 1
                replace2 = 0
            continue
        img_clone[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1] = img[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1]
        img_clone[:,(temp_x * 16):((temp_x+1)* 16):1, (temp_y* 16):((temp_y+1)* 16):1] = img[:,(replace1 * 16):((replace1+1)* 16):1, (replace2* 16):((replace2+1)* 16):1]
        replace2 +=1 
        if(replace2 > 15):
            replace1 += 1
            replace2 = 0
    '''
    plt.subplot(3,1,1)
    plt.imshow(np.asarray(img.cpu()).transpose(1,2,0))      
    plt.subplot(3,1,2)
    plt.imshow(np.asarray(img_clone.cpu()).transpose(1,2,0))
    plt.subplot(3,1,3)
    plt.imshow( abs(np.asarray(img_clone.cpu()).transpose(1,2,0) - np.asarray(img.cpu()).transpose(1,2,0)) )
    plt.show()        
    '''
    return img_clone

def compute_size(channel, group, seed=1):
    np.random.seed(seed)
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    #out = np.random.permutation(out)
    return out


def compute_densemask(in_channels, out_channels, group_num, adj):
    repeat_in = compute_size(in_channels, group_num)
    repeat_out = compute_size(out_channels, group_num)
    mask = adj
    node_mask = mask
    mask = np.repeat(mask, repeat_out, axis=0)
    mask = np.repeat(mask, repeat_in, axis=1)
    return mask, node_mask, repeat_in, repeat_out


def get_mask(in_channels, out_channels, adj):
    group_num, group_num = adj.shape
    assert group_num <= in_channels and group_num <= out_channels
    in_sizes = compute_size(in_channels, group_num)
    out_sizes = compute_size(out_channels, group_num)
    # decide low-level node num
    group_num_low = int(min(np.min(in_sizes), np.min(out_sizes)))
    # decide how to fill each node
    mask_high, node_mask, repeat_in, repeat_out = compute_densemask(in_channels, out_channels, group_num, adj)
    return mask_high, node_mask, repeat_in, repeat_out


def connectGraph(G):
    components = dict(enumerate(nx.connected_components(G)))
    components_combs = combinations(components.keys(), r=2)

    for _, node_edges in groupby(components_combs, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_comps = random.choice(node_edges)
        source = random.choice(list(components[random_comps[0]]))
        target = random.choice(list(components[random_comps[1]]))
        G.add_edge(source, target)
    return G



def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc

def GraphConnectedCheck(adj):
    algorithm = cpnet.BE()
    batchsize, num_patches,num_patches = adj.shape
    cp_graph_cnt = 0
    ave_len_core = []
    for i in range(batchsize):
        G = nx.from_numpy_matrix(adj[i,:,:])
        if( not nx.is_connected(G)):
            G = connectGraph(G)
            temp_adj = nx.to_numpy_array(G)
            adj[i,:,:] = temp_adj
        algorithm.detect(G)
        x = algorithm.get_coreness()
        core_nodes_id = [k for k, v in x.items() if v == 1]
        #print('core_nodes_id ',core_nodes_id)
        ave_len_core.append(len(core_nodes_id))
        #print('len(core_nodes_id) ',len(core_nodes_id))
    print('ave_len_core ', sum(ave_len_core) / len(ave_len_core))
    return adj

'''
def GraphConnectedCheck(adj):
    algorithm = cpnet.BE()
    adj = np.array(adj)
    batchsize, num_patches, num_patches = adj.shape
    cp_graph_cnt = 0
    for i in range(batchsize):
        G = nx.from_numpy_matrix(adj[i,:,:])
        algorithm.detect(G)
        x = algorithm.get_coreness()
        core_nodes_id = [k for k, v in x.items() if v == 1]
        print('core_nodes_id ',core_nodes_id)
        print('len(core_nodes_id) ',len(core_nodes_id))
        if( not nx.is_connected(G)):
            return 0 
    else:
        return 1


def CPGraphGeneration(patchcoreness, option, decisionMargin):
    batchsize,num_heads,_,num_patches = (np.array(patchcoreness.cpu())).shape
    patchcoreness = ( np.sqrt(np.array(patchcoreness.cpu()))).sum(axis = 1)  # batch size * num_heads * 1 * 257 -> batch size * 1 * 257
    patchcoreness = patchcoreness/ patchcoreness.max()
    cp_mask  = np.matmul(np.transpose(patchcoreness,(0,2,1)), patchcoreness)
    if(option == 'soft'):
        cp_mask = np.expand_dims(cp_mask, 1).repeat(num_heads, axis = 1)
        return cp_mask
    else:
        temp = decisionMargin
        while(1):
            temp_cp_mask = copy.deepcopy(cp_mask)
            temp_cp_mask = np.where(temp_cp_mask > temp, 1, 0)
            if(not GraphConnectedCheck(temp_cp_mask)):  #if not connected
                temp = temp - 0.1
            else:
                break
        #print('cp mask threshold ', temp)
        temp_cp_mask = np.expand_dims(temp_cp_mask, 1).repeat(num_heads, axis = 1)
        return temp_cp_mask
'''

def CPGraphGeneration(patchcoreness, option, decisionMargin):
    batchsize,num_heads,_,num_patches = (np.array(patchcoreness.cpu())).shape   #.cpu()
    patchcoreness = ( np.sqrt(np.array(patchcoreness.cpu()))).sum(axis = 1)  #.cpu() batch size * num_heads * 1 * 257 -> batch size * 1 * 257
    batch_atom_max =  patchcoreness.max(axis = 2)
    for i in range(len(batch_atom_max)):
        patchcoreness[i,:,:] = patchcoreness[i,:,:]/batch_atom_max[i,0]
    cp_mask  = np.matmul(np.transpose(patchcoreness,(0,2,1)), patchcoreness)
    if(option == 'soft'):
        cp_mask = np.expand_dims(cp_mask, 1).repeat(num_heads, axis = 1)
        return cp_mask
    else:
        temp = decisionMargin
        temp_cp_mask = copy.deepcopy(cp_mask)
        temp_cp_mask = np.where(temp_cp_mask > temp, 1, 0)
        temp_cp_mask = GraphConnectedCheck(temp_cp_mask)
        temp_cp_mask = np.expand_dims(temp_cp_mask, 1).repeat(num_heads, axis = 1) 
        return temp_cp_mask


def CPGraphGenerationV2(patchcoreness, option, decisionMargin):  #diagnal is ones, i.e., nodes with self-loop
    batchsize,num_heads,_,num_patches = (np.array(patchcoreness.cpu())).shape  #.cpu()
    patchcoreness = (np.sqrt(np.array(patchcoreness.cpu()))) #batch size * num_heads * 1 * 257  .cpu()
    cp_mask  = np.matmul(np.transpose(patchcoreness,(0,1,3,2)), patchcoreness)    #batch size*num_heads*257*1 batch size*num_heads*1*257-> batch size * num_heads * 257 * 257 
    #cp_mask = cp_mask * ( np.ones((batchsize, num_patches, num_patches)) - np.expand_dims(np.eye(num_patches), 0).repeat(batchsize, axis = 0)   ) + np.expand_dims(np.eye(num_patches), 0).repeat(batchsize, axis = 0) 
    if(option == 'soft'):
        #cp_mask = np.expand_dims(cp_mask, 1).repeat(num_heads, axis = 1)       
        return cp_mask
    else:
        temp = decisionMargin
        temp_cp_mask = copy.deepcopy(cp_mask)
        temp_cp_mask = np.where(temp_cp_mask > temp, 1, 0)
        temp_cp_mask = GraphConnectedCheck(temp_cp_mask)
            #if(not GraphConnectedCheck(temp_cp_mask)):  #if not connected
            #    temp = temp - 0.1
            #else:
            #    break
        #print('cp mask threshold ', temp)
        #temp_cp_mask = np.expand_dims(temp_cp_mask, 1).repeat(num_heads, axis = 1)
        return temp_cp_mask


if __name__ == '__main__':
    test_coreness = np.random.rand(2,12,1,257)
    cp_maks = CPGraphGeneration(test_coreness,'hard',0.9)
