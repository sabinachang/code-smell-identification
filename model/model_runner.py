from . import dt
from . import svc
from . import rf
from . import nb
runners = {
    'dt_god': dt.train_god,
    'dt_long': dt.train_long,
    'dt_feature': dt.train_feature,
    'dt_data': dt.train_data,
    'svcl_god': svc.train_linear_god,
    'svcp_god': svc.train_poly_god,
    'svcs_god': svc.train_sigmoid_god,
    'svcr_god': svc.train_rbf_god,
    'svcl_data': svc.train_linear_data,
    'svcp_data': svc.train_poly_data,
    'svcs_data': svc.train_sigmoid_data,
    'svcr_data': svc.train_rbf_data,
    'svcl_long': svc.train_linear_long,
    'svcp_long': svc.train_poly_long,
    'svcs_long': svc.train_sigmoid_long,
    'svcr_long': svc.train_rbf_long,
    'svcl_feature': svc.train_linear_feature,
    'svcp_feature': svc.train_poly_feature,
    'svcs_feature': svc.train_sigmoid_feature,
    'svcr_feature': svc.train_rbf_feature,
    'rf_long': rf.train_long,
    'rf_feature': rf.train_feature,
    'rf_data': rf.train_data,
    'rf_god': rf.train_god,
    'nb_god': nb.train_god,
    'nb_feature': nb.train_feature,
    'nb_long': nb.train_long,
    'nb_data': nb.train_data
}