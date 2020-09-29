from prometheus_http_client import Prometheus
import pandas as pd
import numpy as np
import datetime
import logging
import json
from http.client import HTTPException
import requests

AVG_INT = 15
STEP = 15
MAX_RES = 11000

class agent():

    def __init__(self, name, addr, iface):
        self.name = name
        self.addr = addr
        self.iface = iface

def send_query(query, start, end, step):
    
    prometheus = Prometheus()
    prometheus.url = 'https://thanos.nautilus.optiputer.net'

    res = prometheus.query_rang(metric=query, start=start, end=end, step=step)    
    return res

def prettify_header(metric):
    metrics_to_remove = ['instance', 'job', 'mode', '__name__', 'container', 'endpoint', 'namespace', 'pod', 'prometheus', 'service']
    for i in metrics_to_remove:
        if i in metric: del metric[i]
    if len(metric) > 1 : raise Exception('too many metric labels')
    else:
        return next(iter(metric.keys()))

def export_data_for_prp(sender, receiver, start_time, end_time, **params):
    AVG_INT = 1
    #sender_instance = 'ps-100g.sdsu.edu'
    sender_instance = 'siderea.ucsc.edu'
    receiver_instance = 'k8s-nvme-01.ultralight.org'
    query = (#'label_replace((sum by (container)(rate(container_network_transmit_bytes_total{{namespace=~"{3}", interface="{2}", pod=~"{0}.*"}}[{1}m]) * 8)), "network_throughput", "$0", "container", "(.+)") '
    'label_replace(sum by (instance)(irate(node_network_transmit_bytes_total{{instance=~"{4}.*", device="{2}"}}[{1}m])), "network_throughput", "$0", "instance", "(.+)") '
    'or label_replace(sum by (job)(irate(node_disk_written_bytes_total{{instance=~"{5}.*", device=~"nvme[0-7]n1"}}[{1}m])),"Goodput", "$0", "job", "(.+)") '
    #'or label_replace(sum by (container)(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_rate{{namespace=~"{3}", container="{0}"}}), "CPU", "$0", "container", "(.+)") '
    'or label_replace(sum by (job)(1 - irate(node_cpu_seconds_total{{mode="idle", instance="{4}"}}[1m])),"CPU", "$0", "job", "(.+)") '
    'or label_replace(max by (container)(container_memory_working_set_bytes{{namespace="{3}", container=~"{0}.*"}}), "Memory_used", "$0", "container", "(.+)") '
    'or label_replace(sum by (job)(irate(node_disk_written_bytes_total{{instance=~"{4}.*", device=~"nvme[0-7]n1"}}[{1}m])),"NVMe_from_ceph", "$0", "job", "(.+)") '
    'or label_replace(sum by (job)(irate(node_disk_read_bytes_total{{instance=~"{4}.*", device=~"nvme[0-7]n1"}}[{1}m])),"NVMe_from_transfer", "$0", "job", "(.+)") '
    'or label_replace(sum by (job)(irate(node_disk_io_time_seconds_total{{instance=~"{4}.*", device=~"nvme[0-7]n1"}}[{1}m])),"NVMe_total_util", "$0", "job", "(.+)") '
    'or label_replace(sum by (container)(kube_pod_container_resource_limits_cpu_cores{{container="{0}"}}),"CPU_count", "$0", "container", "(.+)")  '
    'or label_replace(max by (container)(kube_pod_container_resource_limits_memory_bytes{{namespace="{3}", container=~"{0}"}}), "Memory_total", "$0", "container", "(.+)") '
    'or label_replace(count by (job)(node_disk_io_time_seconds_total{{instance=~"{4}.*", device=~"nvme[0-7]n1"}}),"Storage_count", "$0", "job", "(.+)") '
    'or label_replace(sum by (job)(node_network_speed_bytes{{instance=~"{4}.*", device="{2}"}} * 8), "NIC_speed", "$0", "job", "(.+)") '
    '').format(sender['name'], AVG_INT, sender['interface'], 'dtnaas', sender_instance, receiver_instance)
    rows = []
    dataset = None    

    while end_time > start_time:        
        data_in_period = None
        max_ts = start_time + (STEP * MAX_RES)
        # begin = start_time + 100*5
        next_hop_ts = end_time if max_ts > end_time else max_ts
        logging.debug('Getting data for {} : {}'.format(start_time, end_time))
        res = send_query(query, start_time, next_hop_ts, STEP)
        # res = send_query(query, start_time, max_ts, STEP)
        if '401 Authorization Required' in res: raise HTTPException(res)
        response = json.loads(res)
        if response['status'] != 'success': raise Exception('Failed to query Prometheus server')
        
        for result in response['data']['result']:
            result['metric'] = prettify_header(result['metric'])            
            df = pd.DataFrame(data=result['values'], columns = ['Time', result['metric']], dtype=float)            
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.set_index('Time', inplace=True)

            data_in_period = df if data_in_period is None else data_in_period.merge(df, how='outer',  on='Time').sort_index()
        
        dataset = data_in_period if dataset is None else dataset.append(data_in_period)
        start_time = next_hop_ts

    for k,v in params.items():
        dataset[k] = v

    return dataset


def main(transfer_id):
    orchestrator = 'dtn-orchestrator.nautilus.optiputer.net'

    response = requests.get('https://{}/transfer/{}'.format(orchestrator, transfer_id))
    result = response.json()

    sender_id = result['sender']
    receiver_id = result['receiver']

    sender = requests.get('http://{}/DTN/{}'.format(orchestrator, sender_id)).json()
    receiver = requests.get('http://{}/DTN/{}'.format(orchestrator, receiver_id)).json()
    
    ts_data = export_data_for_prp(sender, receiver, result['start_time'], result['end_time'], num_workers = result['num_workers'])

    cols = ts_data.columns.tolist()
    labels_to_rearrange = ['NVMe_total_util', 'NVMe_from_transfer', 'NVMe_from_ceph']    
    for i in labels_to_rearrange: 
        cols.remove(i)
        cols.insert(0,i)    
    
    return ts_data[cols]

if __name__ == '__main__':

    #TODO : Change transfer id to what you want
    for i in range(617,618):
        transfer_id = i
        main(transfer_id).to_csv(str(transfer_id) + '.csv')
    # transfer_id = 591
    # main(transfer_id).to_csv(str(transfer_id) + '.csv')