import requests


orchestrator = 'dtn-orchestrator.nautilus.optiputer.net'
sender_man_addr = '128.114.109.76:5000'
sender_data_addr = '128.114.109.76'
receiver_man_addr = '128.114.109.76:5000'
receiver_data_addr = '128.114.109.76'

transfer_id = None

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
    print(result)
    
    data = {
        'name': 'dtnaas-sender',
        'man_addr': sender_man_addr,
        'data_addr': sender_data_addr,
        'username': 'nobody',
        'interface': 'enp148s0'
    }
    response = requests.post('http://{}/DTN/'.format(orchestrator), json=data)
    result = response.json()
    print(result)


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


def test_transfer():    
    global transfer_id
    result =  requests.get('{}/files'.format('https://dtn-sender.nautilus.optiputer.net'))
    files = result.json()
    print(files)
    file_list = [i['name'] for i in files if i['type'] == 'file']
    # print(file_list)
    dirs = [i['name'] for i in files if i['type'] == 'dir']
    # print(dirs)

    response = requests.post('http://{}/create_dir/'.format('dtn-receiver.nautilus.optiputer.net'),json=dirs)
    # result = response.json()
    # if response.status_code != 200: raise Exception('failed to create dirs')
    print(response)

    # TODO: arrange num workers
    num_workers = 8

    data = {
        'srcfile' : file_list, # [file_list[0]]
        'dstfile' : file_list, # [file_list[0]]
        'num_workers' : num_workers
    }

    response = requests.post('http://{}/transfer/nuttcp/2/1'.format(orchestrator),json=data) # error out
    result = response.json()
    print(result)
    assert result['result'] == True
    transfer_id = result['transfer']
    print('transfer_id %s' % transfer_id )

def wait_for_transfer():
    
    response = requests.post('http://{}/wait/{}'.format(orchestrator, transfer_id))
    result = response.json()
    print(result)


def get_transfer():
    
    response = requests.get('http://{}/transfer/{}'.format(orchestrator, transfer_id))
    result = response.json()
    print(result)


# test_add_DTN()
# already added

# test_ping()
# test_create_file()
test_add_DTN()
test_ping()
test_create_file()
test_transfer()
wait_for_transfer()
get_transfer()