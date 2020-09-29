from datetime import datetime, timedelta
import requests
import time
import pandas
import numpy
import extract
import real_time
import N_XGBoost
import pandas as pd
import os

orchestrator = 'dtn-orchestrator.nautilus.optiputer.net'
# ps-100G
# sender_man_addr = '130.191.103.1:5000'
# sender_data_addr = '130.191.103.1'

sender_man_addr = '128.114.109.76:5000'
sender_data_addr = '128.114.109.76'

receiver_man_addr = '198.32.43.89:5000'
receiver_data_addr = '198.32.43.89'

def test_add_DTN():
    
    data = {
        'name': 'dtnaas-receiver',
        'man_addr': receiver_man_addr,
        'data_addr': receiver_data_addr,
        'username': 'nobody',
        'interface': 'ens2'
    }
    response = requests.post('http://{}/DTN/'.format(orchestrator), json=data)    
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
    response = requests.post('http://{}/DTN/'.format(orchestrator), json=data)
    result = response.json()
    assert result == {'id': 2}


def test_ping():
    response = requests.get('http://{}/ping/2/1'.format(orchestrator))
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


def test_transfer(num_workers = 8, parallel = 10):    
    result =  requests.get('{}/files/project'.format('https://dtn-sender.nautilus.optiputer.net'))
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

    response = requests.post('https://{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    print('transfer_id %s, parallel %s' % (transfer_id, parallel) )
    return transfer_id

def test_transfer_with_dynamic_msrsync(num_workers = 8):

    nvme_usage = list(pandas.read_csv('nvme_usage_daily.csv')['mean'])
    iterator = iter(nvme_usage)
    
    result =  requests.get('{}/files/project'.format('https://dtn-sender.nautilus.optiputer.net'))
    files = result.json()
    file_list = ['project/' + i['name'] for i in files if i['type'] == 'file']
    dirs = ['project/' + i['name'] for i in files if i['type'] == 'dir']

    response = requests.post('http://{}/create_dir/'.format('dtn-receiver.nautilus.optiputer.net'),json=dirs)
    if response.status_code != 200: raise Exception('failed to create dirs')

    # data = {            
    #     'address' : '/DMC/lolipop_raw',
    #     'file' : '/data/project2',
    #     'parallel' : parallel
    # }

    # response = requests.post('{}/receiver/msrsync'.format('https://dtn-sender.nautilus.optiputer.net'), json=data)
    # result = response.json()
    # assert result.pop('result') == True
    
    data = {
        'srcfile' : file_list,
        'dstfile' : file_list, #['/dev/null'] * len(file_list),
        'num_workers' : num_workers#,
        #'blocksize' : 8192
    }

    response = requests.post('https://{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    print('transfer_id %s' % transfer_id)

    while requests.get('https://{}/check/{}'.format(orchestrator, transfer_id)).json()['Unfinished'] != 0:

        parallel = int(next(iterator)/ 50)
        print('parallel %s' %parallel)

        data = {
            'address' : '/DMC/lolipop_raw',
            'file' : '/data/project2',
            'parallel' : parallel
        }

        response = requests.post('{}/receiver/msrsync'.format('http://dtn-sender.nautilus.optiputer.net'), json=data)
        result = response.json()
        assert result.pop('result') == True

        time.sleep(60)
        response = requests.get('{}/cleanup/msrsync'.format('http://dtn-sender.nautilus.optiputer.net'))            

    print('Finished')

    return transfer_id

def finish_transfer(transfer_id):    
    response = requests.post('http://{}/wait/{}'.format(orchestrator, transfer_id))
    result = response.json()
    #print(result)

    cleanup()
    
def cleanup(retry = 5):

    for i in range(0, retry):        
        response = requests.get('http://{}/cleanup/nuttcp'.format('dtn-sender.nautilus.optiputer.net'))
        if response.status_code != 200: continue
        response = requests.get('http://{}/cleanup/nuttcp'.format('dtn-receiver.nautilus.optiputer.net'))
        if response.status_code != 200: continue

        response = requests.get('http://{}/cleanup/stress'.format('dtn-sender.nautilus.optiputer.net'))
        #response = requests.get('http://{}/cleanup/msrsync'.format('dtn-sender.nautilus.optiputer.net'))        
        #response = requests.delete('http://{}/file/project2'.format('dtn-sender.nautilus.optiputer.net'))
        
        return 
    raise Exception('Cannot cleanup after %s tries' % retry)

def get_transfer(transfer_id):
    
    response = requests.get('http://{}/transfer/{}'.format(orchestrator, transfer_id))
    result = response.json()
    print(result)

def parse_nvme_usage(filename):
    df = pandas.read_csv(filename, parse_dates=[0])    
    df['elapsed'] =  (df['Time'] - df['Time'][0])/numpy.timedelta64(1,'s')

    df = df[['elapsed', 'mean']].astype('int32').set_index('elapsed')
    df['mean'] = df['mean'].apply(str) + 'M'
    
    return df.to_dict()['mean']

def prepare_transfer():
    result =  requests.get('{}/files/project'.format('https://dtn-sender.nautilus.optiputer.net'))
    files = result.json()
    file_list = ['project/' + i['name'] for i in files if i['type'] == 'file']
    dirs = ['project/' + i['name'] for i in files if i['type'] == 'dir']

    response = requests.post('http://{}/create_dir/'.format('dtn-receiver.nautilus.optiputer.net'),json=dirs)
    if response.status_code != 200: raise Exception('failed to create dirs')
    return file_list

def start_transfer(file_list, num_workers):
    data = {
        'srcfile' : file_list,
        'dstfile' : file_list, #['/dev/null'] * len(file_list),
        'num_workers' : num_workers#,
        #'blocksize' : 8192
    }

    response = requests.post('https://{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    return transfer_id

def start_nvme_usage(nvme_usage):
    data = {
        'sequence' : nvme_usage,         
        'file':'disk0/fiotest',
        'size' : '1G',
        'address' : ''
    }
    response = requests.post('{}/receiver/stress'.format('https://dtn-sender.nautilus.optiputer.net'), json=data)
    result = response.json()
    assert result.pop('result') == True

def wait_for_transfer(transfer_id):
    while True:
        response = requests.get('https://{}/check/{}'.format(orchestrator, transfer_id))
        result = response.json()
        if result['Unfinished'] == 0:
            response = requests.get('http://{}/cleanup/stress'.format('dtn-sender.nautilus.optiputer.net'))
            break
        time.sleep(30)


def test_transfer_with_fio(num_workers):
    nvme_usage = parse_nvme_usage('nvme_usage_daily.csv')

    file_list = prepare_transfer()
    transfer_id = start_transfer(file_list, num_workers)
    
    print('transfer_id %s' % transfer_id)

    start_nvme_usage(nvme_usage)

    wait_for_transfer(transfer_id)

    response = requests.get('{}/stress/poll'.format('https://dtn-sender.nautilus.optiputer.net'), json={})
    #result = response.json()
    #assert response.status_code == 200       

    return transfer_id

def test_dynamic_transfer(num_workers, sequence,Name):
    assert type(sequence) == dict    

    nvme_usage = parse_nvme_usage('nvme_usage_daily.csv')
    sender = requests.get('http://{}/DTN/{}'.format(orchestrator, 2)).json()
    receiver = requests.get('http://{}/DTN/{}'.format(orchestrator, 1)).json()

    file_list = prepare_transfer()
    transfer_id = start_transfer(file_list, num_workers)

    start_time = datetime.now()
    print('transfer_id %s , start_time %s' % (transfer_id, start_time))

    start_nvme_usage(nvme_usage)

    collect_dir = 'D:/N_Prediction/data/real-time/collect/' + str(sequence[0]['num_workers']) + '/'
    folder_name = str(sequence[0]['num_workers']) + '_' + str(Name)
    os.mkdir(os.path.join(collect_dir, folder_name))

    featuer_dir = 'D:/N_Prediction/data/real-time/feature/' + str(sequence[0]['num_workers']) + '/'
    os.mkdir(os.path.join(featuer_dir, folder_name))

    intervals = sorted(sequence.keys())
    for interval in intervals:
        if interval == 0 :
            N = sequence[interval]['num_workers']
            continue
        # sleeping for the next change of N interval
        while interval > (datetime.now() - start_time).total_seconds():
            time.sleep(0.1)

        # getting real-time data from the start of the transfer to now.
        # You can update the sequence as you change dynamically after the transfer/id/scale api call
        print('elapsed_time %s, interval %s' % ((datetime.now() - start_time).total_seconds(), interval))
        dataframe = get_realtime_data(sender, receiver, start_time, sequence)

        dataframe.to_csv(collect_dir+ folder_name +'/'+str(int(N))+'.csv',index=None)

        features = ['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util','num_workers']
        pred = pd.DataFrame(data=dataframe, columns=features)
        print(pred)

        firststarttime = datetime.now()
        real_time.prediction(collect_dir + folder_name +'/' +str(int(N))+'.csv')
        firstendtime = datetime.now()
        firstuse = firstendtime-firststarttime
        print('first predition time:',firstuse.total_seconds(),'s')

        secondstarttime = datetime.now()
        N = N_XGBoost.get_N(featuer_dir + folder_name +'/' +str(int(N))+'_real.csv')
        secondendtime = datetime.now()
        seconduse = secondendtime - secondstarttime
        print('second predition time:', seconduse.total_seconds(), 's')

        # timeusepath = 'D:/N_Prediction/predictiontime2.csv'
        # if not os.path.isfile(timeusepath):
        #     datausetime = pd.DataFrame(columns=['firsttime','secondtime'],data=[[firstuse,seconduse]])
        # else:
        #     datausetime = pd.read_csv(timeusepath)
        #     datausetime = datausetime.append([{'firsttime': firstuse, 'secondtime': seconduse}], ignore_index=True)
        #
        #
        # datausetime.to_csv(timeusepath, index=None)

        print('num_wokers:' + str(N))
        data = sequence[interval]
        sequence.pop(interval)

        sequence[interval] = {'num_workers' : int(N)}
        data = sequence[interval]
        # data ={'num_wokers': int(N)}
        response = requests.post('https://{}/transfer/{}/scale'.format(orchestrator, transfer_id), json=data)
        # update sequence if it is not from the data.
        if response.status_code != 200:
            print('failed to change transfer parameter')
            # break
        else:
            print('Changed the parameters to %s' % data)

    wait_for_transfer(transfer_id)

    response = requests.get('{}/stress/poll'.format('https://dtn-sender.nautilus.optiputer.net'), json={})
    #result = response.json()
    #assert response.status_code == 200   

    return transfer_id

def get_realtime_data(sender, receiver, start_time, sequence):
    
    df = extract.export_data_for_prp(sender, receiver, start_time.timestamp(), datetime.now().timestamp())
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
    #test_add_DTN()
    # already added

    #test_ping()
    # test_create_file()
    # 0: {'num_workers': 8},
    # 120: {'num_workers': 55},
    # 240: {'num_workers': 67},
    # 480: {'num_workers': 58},
    # 600: {'num_workers': 70},
    # 720: {'num_workers': 58},
    # 840: {'num_workers': 58}
    sequence = { #time : parameters. Need to be updated dynamically for real-time data extraction
            0 : {'num_workers' : 40},
            120: {'num_workers': 40},
            240: {'num_workers': 40},
            360: {'num_workers': 40},
            480: {'num_workers': 40},
            600: {'num_workers': 40},


        }

    cleanup()
    for i in range(16,17):
        transfer_id = test_dynamic_transfer(sequence[0]['num_workers'], sequence,i)
        finish_transfer(transfer_id)
        get_transfer(transfer_id)
        print(str(sequence))
        sequence = {  # time : parameters. Need to be updated dynamically for real-time data extraction
            0: {'num_workers': 40},
            120: {'num_workers': 40},
            240: {'num_workers': 40},
            360: {'num_workers': 40},
            480: {'num_workers': 40},
            600: {'num_workers': 40},

        }
