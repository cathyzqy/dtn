from datetime import datetime, timedelta
import requests
import time
import pandas
import numpy
import extractmrp
import random
import real_time
import N_XGBoost


def test_add_DTN(cluster = 'prp'):
    
    if cluster == 'prp':
        sender_man_addr = '128.114.109.76:5000'
        sender_data_addr = '128.114.109.76'

        receiver_man_addr = '198.32.43.89:5000'
        receiver_data_addr = '198.32.43.89'   

        data = {
            'name': 'dtnaas-receiver',
            'man_addr': receiver_man_addr,
            'data_addr': receiver_data_addr,
            'username': 'nobody',
            'interface': 'ens2'
        }
        response = requests.post('{}/DTN/'.format(orchestrator), json=data)    
        result = response.json()
        assert result == {'id': 1}
        
        data = {
            'name': 'dtnaas-sender',
            'man_addr': sender_man_addr,
            'data_addr': sender_data_addr,
            'username': 'nobody',
            # 'interface': 'enp148s0'
            'interface': 'ens2'
        }
        response = requests.post('{}/DTN/'.format(orchestrator), json=data)
        result = response.json()
        assert result == {'id': 2}

    elif cluster == 'mrp':
        sender_man_addr = '74.114.96.103:5000'
        sender_data_addr = '74.114.96.103'

        receiver_man_addr = '131.193.183.248:5000'
        receiver_data_addr = '131.193.183.248'

        data = {
            'name': 'dtnaas-receiver',
            'man_addr': receiver_man_addr,
            'data_addr': receiver_data_addr,
            'username': 'nobody',
            'interface': 'enp8s0'
        }
        response = requests.post('{}/DTN/'.format(orchestrator), json=data)    
        result = response.json()
        assert result == {'id': 1}
        
        data = {
            'name': 'dtnaas-sender',
            'man_addr': sender_man_addr,
            'data_addr': sender_data_addr,
            'username': 'nobody',            
            'interface': 'vlan555'
        }
        response = requests.post('{}/DTN/'.format(orchestrator), json=data)
        result = response.json()
        assert result == {'id': 2}

    else:
        raise Exception('Only prp and mrp is supported')


def test_ping(orchestrator):
    response = requests.get('{}/ping/2/1'.format(orchestrator))
    result = response.json()
    print(result)


def test_create_file():
    data = {
            'hello_world': {                
                'size': '100G'
            },
            'hello_world2': {
                'size': '100G'
            },
            'hello_world3': {
                'size': '100G'
            },
            'hello_world4': {
                'size': '100G'
            }
        }
    response = requests.post('http://{}/create_file/'.format('dtn-sender.nautilus.optiputer.net'), json=data)
    result = response.json()
    print(result)


def test_transfer(orchestrator, num_workers = 8, parallel = 10, ):    
    result =  requests.get('{}/files/project'.format('http://dtn-sender.starlight.northwestern.edu'))
    files = result.json()
    file_list = ['project/' + i['name'] for i in files if i['type'] == 'file']
    dirs = ['project/' + i['name'] for i in files if i['type'] == 'dir']

    response = requests.post('http://{}/create_dir/'.format('dtn-receiver.nautilus.optiputer.net'),json=dirs)
    if response.status_code != 200: raise Exception('failed to create dirs')

    data = {            
        'address' : '/DMC/lolipop_raw',
        'file' : '/data/project2',
        'parallel' : parallel
    }

    response = requests.post('{}/receiver/msrsync'.format('https://dtn-sender.nautilus.optiputer.net'), json=data)
    result = response.json()
    assert result.pop('result') == True
    
    data = {
        'srcfile' : file_list,
        'dstfile' : file_list, #['/dev/null'] * len(file_list),
        'num_workers' : num_workers#,
        #'blocksize' : 8192
    }

    response = requests.post('{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    print('transfer_id %s, parallel %s' % (transfer_id, parallel) )
    return transfer_id

def finish_transfer(transfer_id, orchestrator, sender, receiver):    
    response = requests.post('{}/wait/{}'.format(orchestrator, transfer_id))
    result = response.json()
    #print(result)

    cleanup(sender, receiver)
    
def cleanup(sender, receiver, retry = 5):

    for i in range(0, retry):        
        response = requests.get('{}/cleanup/nuttcp'.format(sender))
        if response.status_code != 200: continue
        response = requests.get('{}/cleanup/nuttcp'.format(receiver))
        if response.status_code != 200: continue

        response = requests.get('{}/cleanup/stress'.format(sender))
        
        return 
    raise Exception('Cannot cleanup after %s tries' % retry)

def get_transfer(transfer_id, orchestrator):
    
    response = requests.get('{}/transfer/{}'.format(orchestrator, transfer_id))
    result = response.json()
    print(result)

def parse_nvme_usage(filename):
    df = pandas.read_csv(filename, parse_dates=[0])    
    df['elapsed'] =  (df['Time'] - df['Time'][0])/numpy.timedelta64(1,'s')

    df = df[['elapsed', 'mean']].astype('int32').set_index('elapsed')
    df['mean'] = df['mean'].apply(str) + 'M'
    
    return df.to_dict()['mean']

def prepare_transfer(srcdir, sender, receiver):
    # result =  requests.get('{}/files/{}'.format(sender, srcdir))
    result = requests.get('{}/files/project'.format('https://dtn-sender.nautilus.optiputer.net'))
    files = result.json()
    file_list = [srcdir + i['name'] for i in files if i['type'] == 'file']
    dirs = [srcdir + i['name'] for i in files if i['type'] == 'dir']

    response = requests.post('{}/create_dir/'.format(receiver),json=dirs)
    if response.status_code != 200: raise Exception('failed to create dirs')
    return file_list

def start_transfer(file_list, num_workers, orchestrator):
    data = {
        'srcfile' : file_list,
        'dstfile' : file_list, #['/dev/null'] * len(file_list),
        'num_workers' : num_workers#,
        #'blocksize' : 8192
    }

    response = requests.post('{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    return transfer_id

def start_nvme_usage(nvme_usage, sender):
    data = {
        'sequence' : nvme_usage,         
        'file':'disk0/fiotest',
        'size' : '1G',
        'address' : ''
    }
    response = requests.post('{}/receiver/stress'.format(sender), json=data)
    result = response.json()
    assert result.pop('result') == True

def wait_for_transfer(transfer_id, orchestrator, sender):
    while True:
        response = requests.get('{}/check/{}'.format(orchestrator, transfer_id))
        result = response.json()
        if result['Unfinished'] == 0:
            response = requests.get('{}/cleanup/stress'.format(sender))
            break
        time.sleep(30)


def test_transfer_with_fio(num_workers, sender, receiver, srcdir):
    nvme_usage = parse_nvme_usage('nvme_usage_daily.csv')

    file_list = prepare_transfer(srcdir, sender, receiver)
    transfer_id = start_transfer(file_list, num_workers, orchestrator)
    
    print('transfer_id %s' % transfer_id)

    start_nvme_usage(nvme_usage)

    wait_for_transfer(transfer_id,orchestrator, sender)

    response = requests.get('{}/stress/poll'.format(sender), json={})
    #result = response.json()
    #assert response.status_code == 200       

    return transfer_id

def test_dynamic_transfer(num_workers, sequence, orchestrator, sender, receiver, sender_instance, receiver_instance, srcdir, monitor, cluster):
    assert type(sequence) == dict    

    nvme_usage = parse_nvme_usage('nvme_usage_daily.csv')    
    sender_obj = requests.get('{}/DTN/{}'.format(orchestrator, 2)).json()
    receiver_obj = requests.get('{}/DTN/{}'.format(orchestrator, 1)).json()

    file_list = prepare_transfer(srcdir, sender, receiver)
    transfer_id = start_transfer(file_list, num_workers, orchestrator)

    start_time = datetime.now()
    print('transfer_id %s , start_time %s' % (transfer_id, start_time))

    start_nvme_usage(nvme_usage, sender)

    intervals = sorted(sequence.keys())
    for interval in intervals:
        if interval == 0 :
            N = sequence[interval]['num_workers']
            # N1 = 50
            # N2 = 50
            # N3 = 50
            # N4 = 50
            # N5 = 50
            # N6 = 50
            continue
        # sleeping for the next change of N interval
        while interval > (datetime.now() - start_time).total_seconds():             
            time.sleep(0.1)

        # getting real-time data from the start of the transfer to now.
        # You can update the sequence as you change dynamically after the transfer/id/scale api call
        print('elapsed_time %s, interval %s' % ((datetime.now() - start_time).total_seconds(), interval))
        dataframe = get_realtime_data(sender_obj, receiver_obj, sender_instance, receiver_instance, start_time, sequence, monitor, cluster)
        dataframe.to_csv('D:/N_Prediction/data/real-time/collect/' + str(int(N)) + '.csv',index=None)
        print(dataframe)
        real_time.prediction('D:/N_Prediction/data/real-time/collect/' + str(int(N)) + '.csv')
        N = N_XGBoost.get_N('D:/N_Prediction/data/real-time/feature/' + str(int(N)) + '_real.csv')
        print('num_wokers:' + str(N))


        data = sequence[interval]
        sequence.pop(interval)

        sequence[interval] = {'num_workers' : int(N)}
        data = sequence[interval]
        response = requests.post('{}/transfer/{}/scale'.format(orchestrator, transfer_id), json=data)
        # update sequence if it is not from the data.
        if response.status_code != 200:
            print('failed to change transfer parameter')
            break
        else:
            print('Changed the parameters to %s' % data)

    wait_for_transfer(transfer_id, orchestrator, sender)

    response = requests.get('{}/stress/poll'.format(sender), json={})
    #result = response.json()
    #assert response.status_code == 200   

    return transfer_id

def get_realtime_data(sender, receiver, sender_instance, receiver_instance, start_time, sequence, monitor, cluster):
    
    df = extractmrp.export_data_for_prp(sender, receiver, sender_instance, receiver_instance, start_time.timestamp(), datetime.now().timestamp(), monitor, cluster )
    df = update_params(df, sequence)

    return df

def update_params(df, sequence):

    indices = sorted(sequence.keys())    
    for i in range(0, len(indices)):
        next_t = indices[i+1] if i < len(indices) - 1 else (df.index[-1] - df.index[0]).seconds
        for j in sequence[indices[i]].keys():
            if j not in df:
                df[j] = numpy.NaN
            df.loc[((df.index - df.index[0]).seconds >= indices[i]) & ((df.index - df.index[0]).seconds < next_t), j] = sequence[indices[i]][j]        
    return df

    
if __name__ == "__main__":

    cluster = 'mrp'

    if cluster == 'prp':
        orchestrator = 'https://dtn-orchestrator.nautilus.optiputer.net'
        sender = 'http://dtn-sender.nautilus.optiputer.net'
        receiver = 'http://dtn-receiver.nautilus.optiputer.net'
        srcdir = 'project/'
        sender_instance = 'siderea.ucsc.edu'
        receiver_instance = 'k8s-nvme-01.ultralight.org'
        monitor = 'https://thanos.nautilus.optiputer.net'
    elif cluster == 'mrp':
        orchestrator = 'http://dtn-orchestrator.starlight.northwestern.edu'
        sender = 'http://dtn-sender.starlight.northwestern.edu'
        receiver = 'http://dtn-receiver.starlight.northwestern.edu'
        srcdir = 'project/'
        sender_instance = '165.124.33.175:9100'
        receiver_instance = '131.193.183.248:9100'
        monitor = 'http://165.124.33.158:9091/'
    else: raise Exception('Only prp or mrp is supported')

    # already added
    # test_add_DTN(cluster=cluster)
    # test_ping(orchestrator)    

    # sequence = { #time : parameters. Need to be updated dynamically for real-time data extraction
    #     0 : {'num_workers' : 70},
    #     120 : {'num_workers' : 70},
    #     240 : {'num_workers' : 70},
    #     360 : {'num_workers' : 70},
    #     480: {'num_workers': 70},
    #     600: {'num_workers': 70},
    #     # 720: {'num_workers': 50},
    #     # 840: {'num_workers': 58},
    # }
    sequence = {
        0: {'num_workers': 20},
        120: {'num_workers': 20},
        240: {'num_workers': 20},
        360: {'num_workers': 20},
        480: {'num_workers': 20},
        # 600: {'num_workers': 50},
    }

    cleanup(sender, receiver)

    transfer_id = test_dynamic_transfer(sequence[0]['num_workers'], sequence, orchestrator, sender, receiver, sender_instance, receiver_instance, srcdir, monitor, cluster)
    finish_transfer(transfer_id, orchestrator, sender, receiver)
    get_transfer(transfer_id, orchestrator)
    print(sequence)
