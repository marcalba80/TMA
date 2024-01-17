import pyshark
import math
from datetime import datetime, timedelta
import csv
import sys
import os

def get_domain_features(string):
    upper = 0
    lower = 0
    numeric = 0
    special = 0

    char_frequency = {}

    for c in string:
        if c.isalpha():
            if c.isupper():
                upper += 1
            else:
                lower += 1
        elif c.isdigit():
            numeric += 1
        else:
            special += 1

        # For entropy:
        if c in char_frequency:
            char_frequency[c] += 1
        else:
            char_frequency[c] = 1

    entropy = 0
    for char, freq in char_frequency.items():
        prob = freq / len(string)
        entropy -= prob * math.log2(prob)

    return upper, lower, numeric, special, entropy


PCAP_FILENAME = sys.argv[1]
filename, _ = os.path.splitext(PCAP_FILENAME)
CSV_FILENAME = f'{filename}_features.csv'

#PCAP_FILENAME = "benign_1.pcap"
#CSV_FILENAME = "benign_1_features.csv"

#PCAP_FILENAME = "sample.pcapng"
#CSV_FILENAME = "sample_features.csv"


# STATEFUL FEATURES
# rr  A_frequency  NS_frequency  CNAME_frequency  SOA_frequency  NULL_frequency  PTR_frequency  HINFO_frequency  MX_frequency  TXT_frequency  
# AAAA_frequency  SRV_frequency  OPT_frequency  rr_type  rr_count  +rr_name_entropy  +rr_name_length  distinct_ns  distinct_ip  +unique_country
# unique_asn  distinct_domains  reverse_dns  a_records  unique_ttl  ttl_mean  ttl_variance
# New stateful features: time difference, following tcp connection, how many queries to the same domain have we previously seen

# 0.0,0,0,0,0,0,0,0,0,0,0,0,0,{None},0,3.4639847730197832,32,0,set(),set(),set(),{},unknown,0,"[64, 64, 64, 64]",64.0,0.0

# STATELESS FEATURES
# timestamp  FQDN_count  subdomain_length  upper  lower  numeric  entropy  special  labels  labels_max  labels_average  +longest_word  +sld  len  subdomain
# 2024-01-15 18:13:11,        27, 14, 0,  23, 0,  3.6619328723735824, 4, 5, 7, 4.6,                0, 0,   23, 1 OK
# 2020-11-21 19:13:27.034607, 27, 10, 0,  10, 11, 2.5704170701729945, 6, 6, 7, 3.6666666666666665, 2, 192, 14, 1


pcap = pyshark.FileCapture(PCAP_FILENAME)

dict = {}
packetsByDomain = {}
ipsToDomain = {}

data = {}
data[0] = ["timestamp,FQDN_count,subdomain_length,upper,lower,numeric,entropy,special,labels,labels_max,labels_average,longest_word,sld,len,subdomain,time_difference,packetsToSameDomain"]

print("Generating features...")

packetNumber = 1
start = datetime.now()
for packet in pcap:
    if 'DNS' in packet:
        try:
            domain = packet.dns.qry_name

            timestamp = datetime.fromtimestamp(float(packet.sniff_timestamp))
            timeSinceLastPacket = 0
            packetsToSameDomain = 0

            # Add packet to the list
            if domain not in packetsByDomain:
                packetsByDomain[domain] = {}
            else:
                last_key = max(packetsByDomain[domain].keys())
                previousTimestamp = datetime.fromtimestamp(float(packetsByDomain[domain][last_key].sniff_timestamp))
                timeSinceLastPacket = timestamp - previousTimestamp
                packetsToSameDomain = len(packetsByDomain[domain])

                
            packetsByDomain[domain][packetNumber] = packet

            FQDN_count = len(domain)

            subdomain_length = 0
            subdomain = 0
            labels = domain.split('.')
            if len(labels) > 2:
                subdomain_name = '.'.join(labels[:2])
                subdomain_length = len(subdomain_name)
                subdomain = 1
            elif len(labels) == 2:
                subdomain_length = len(labels[0])
                subdomain = 1

            labels_max = max(labels, key=len)
            _len = sum(len(label) for label in labels)
            labels_average = _len / len(labels)

            upper, lower, numeric, special, entropy = get_domain_features(domain)

            #print(f'Number {packetNumber} with timestamp {timestamp} - domain: {domain}, upper: {upper}, lower: {lower}, numeric: {numeric}, special: {special}, entropy: {entropy}, labels_max = {labels_max}, timeSinceLastPacket = {timeSinceLastPacket}, queriesToSameDomain = {packetsToSameDomain}')
            data[packetNumber] = [f'{timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")},{FQDN_count},{subdomain_length},{upper},{lower},{numeric},{entropy},{special},{len(labels)},{len(labels_max)},{labels_average},0,0,{_len},{subdomain},{timeSinceLastPacket},{packetsToSameDomain},0']
        
            if packet.dns.flags_response == 'True': # DNS RESPONSE
                try:
                    ipsToDomain[packet.dns.a] = domain
                except Exception:
                    packetNumber +=1
                    continue
        except Exception:
            continue
            
            
            
            
            #if packetNumber == 258:
            #    print(f'{packet}')
#
#
            #    print(f'Packet info: resp_name = {packet.dns.resp_name}')
            #        
            #    print(f'Packet info: resp_type = {packet.dns.resp_type}')
            #    
            #    print(f'Packet info: a = {packet.dns.a}')
            #    
            #    for i in range(int(packet.dns.count_answers)):
            #        try:
            #            if packet.dns.resp_type == '1':  # Type 1 is A record
            #                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            #                ip_address = packet.dns['dns.a'][i]
            #                print(f"Resolved IP Address: {ip_address}")
            #        except AttributeError:
            #            break
            #        except Exception:
            #            print(f'ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR!!!!! {Exception}')
            #            break

    # Calculating TCP connection feature:
    # 3 MB File of normal traffic mixed with DNS traffic:
    # From 8.3 to 8.5 seconds

    elif 'TCP' in packet:
        is_syn = packet.tcp.flags_syn == 'True'
        is_ack = packet.tcp.flags_ack == 'True'

        if is_syn and not is_ack: # TCP Initiator
            suspicious_domain = ""
            if packet.ip.src in ipsToDomain:
                suspicious_domain = ipsToDomain[packet.ip.src]
            elif packet.ip.dst in ipsToDomain:
                suspicious_domain = ipsToDomain[packet.ip.dst]
            
            if suspicious_domain in packetsByDomain:
                tcpTimestamp = datetime.fromtimestamp(float(packet.sniff_timestamp))
                limitTimestamp = tcpTimestamp - timedelta(seconds=1)
                for key in reversed(packetsByDomain[suspicious_domain].keys()):
                    packetTimestamp = datetime.fromtimestamp(float(packetsByDomain[suspicious_domain][key].sniff_timestamp))
                    if  packetTimestamp >= limitTimestamp:
                        data[key][-1] = data[key][-1][:-1] + '1'
                    else:
                        break


    # End of loop
    packetNumber +=1
pcap.close()

#print('IPS-DOMAIN MAP:')
#for key in ipsToDomain.keys():
#    print(f'IP {key} associated to domain {ipsToDomain[key]}')
#
#print(f'Features generated.')

end = datetime.now()

print(f'Took {end - start} seconds')


#print('Generating stateful features...')      
#for packetNumber, packet in DNS_packets.items():
#    statefulFeatureList = []
#    domain = packet.dns.qry_name
#    record_name = ""
#    A_frequency = NS_frequency = CNAME_frequency = SOA_frequency = NULL_frequency = PTR_frequency = HINFO_frequency = MX_frequency = TXT_frequency = AAAA_frequency = SRV_frequency = OPT_frequency = 0
#    
#    if packetNumber == 258:
#        print(f'Packet 258: {packet}')
#    
#    if packet.dns.flags_response == 'False': 
#        record_name = domain            
#        
#    else: #RESPONSE
#        num_answers = int(packet.dns.count_answers)
#        for i in range(num_answers):
#            #print(f'AAAAAAAAAAAAAAAAAAAAA {packet.dns.record}')
#            #resp_name = getattr(packet.dns, f'dns_resp_name_{i}')
#            #resp_type = getattr(packet.dns, f'dns_resp_type_{i}')
#            print(f'Packet {packetNumber}: DNS response record name number {i}: {record_name} of type {resp_type}')
#

                 

print(f'Writing file {CSV_FILENAME}...')

with open(CSV_FILENAME, 'w', newline='') as file:
    writer=csv.writer(file)

    for key in data.keys():
        writer.writerow(data[key])

print(f'File {CSV_FILENAME} generated.')