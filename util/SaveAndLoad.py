import numpy as np
import os

from distributions.lin_conditional.LinMOE import LinMOE

def save_model_linmoe(model, save2path, it=None):

    means_cmps = np.stack([c.params for c in model.components], axis=0)
    covars_cmps = np.stack([c.covar for c in model.components], axis=0)

    means_ctxt_cmps = np.stack([c.mean for c in model.ctxt_components], axis=0)
    covars_ctxt_cmps = np.stack([c.covar for c in model.ctxt_components], axis=0)

    weight_distr_probs = model.weight_distribution.probabilities

    model_dict = {'weights_distr_probs': weight_distr_probs, 'means_ctxt_cmps': means_ctxt_cmps,
                  'covars_ctxt_cmps': covars_ctxt_cmps, 'means_cmps': means_cmps, 'covars_cmps':covars_cmps}
    filename = 'linmoe_model'
    if it is None:
        savepath = os.path.join(save2path, filename + '.npz')
    else:
        savepath = os.path.join(save2path, filename + '_' + str(it)+'.npz')

    np.savez_compressed(savepath, **model_dict)

def load_model_linmoe(path2load, it=None):
    filename='linmoe_model'
    if it is None:
        model_path = os.path.join(path2load, filename + '.npz')
    else:
        model_path = os.path.join(path2load, filename + '_' + str(it) + '.npz')
    model_dict = dict(np.load(model_path, allow_pickle=True))
    model = LinMOE(cmp_params=model_dict['means_cmps'], cmp_covars=model_dict['covars_cmps'], ctxt_cmp_means=model_dict['means_ctxt_cmps'],
                   ctxt_cmp_covars=model_dict['covars_ctxt_cmps'])
    model.weight_distribution._p = model_dict['weights_distr_probs']
    return model

def load_and_sort_all_models_linmoe(path2load):
    all_data = os.listdir(path2load)
    models = {}
    last_it=100000
    for name in all_data:
        if name.split('.')[1] == 'npz':
            it = name.split('_')[-1].split('.')[0]
            if it not in ['e', 'c', 'entropy', 'reward', 'config', 'model', 'interacts', 'executed']:
                models[int(it)] = load_model_linmoe(path2load, it=int(it))
    try:
        models[last_it] = load_model_linmoe(path2load)
    except:
        print('last model was not saved!')
    keys_list = list(models.keys())
    sorted_iterations_idx = np.argsort(keys_list)
    sorted_keys_np_arr = np.array(keys_list)[sorted_iterations_idx]
    sorted_models = []
    for it in sorted_iterations_idx:
        c_key = keys_list[it]
        c_model = models[c_key]
        sorted_models.append(c_model)

    return models, sorted_models, sorted_keys_np_arr

def save_model_lin_emm(model, save2path, quad_feats=False, it=None):

    # components
    means_comps = np.stack([c.params for c in model.components], axis=0)
    covars_comps = np.stack([c.covar for c in model.components], axis=0)

    # softmax
    gating_params = model.gating_distribution.params

    model_dict = {'gating_params': gating_params, 'means_comps': means_comps, 'covars_comps':covars_comps,
                  'quad_feats': quad_feats}
    if it is None:
        file_name = 'model'
    else:
        file_name = 'model_it_' + str(it)
    np.savez_compressed(os.path.join(save2path, file_name + '.npz'), **model_dict)