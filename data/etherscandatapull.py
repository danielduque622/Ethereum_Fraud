import dotenv
import os
from web3 import Web3
import requests
import pandas as pd

dotenv.load_dotenv(".env", override=True)

def formQueryString(address, pgNo, offset, api_key):
    return (
        f"https://api.etherscan.io/api?module=account&action=txlist&address={address}"
        f"&startblock=0&endblock=99999999&page={pgNo}&offset={offset}&sort=asc&apikey={api_key}"
    )

def get_address_stats_normal_tnx(address):
    response = requests.get(formQueryString(address,1,10000,os.environ.get("ETHERSCAN_API_KEY")))

    sample_df = pd.DataFrame(response.json()['result'])
    # Column creation of ETH from Wei
    sample_df['eth value'] = sample_df['value'].apply(lambda x: Web3.from_wei(int(x),'ether'))

    # Typing of sent and received transactions
    sample_df['txn type'] = sample_df['from'].apply(lambda x: 'sent' if x== address else 'received')

    # Handling of Sent transactions stats
    sample_df_sent =  sample_df[sample_df['txn type'] == 'sent']
    sample_df_sent = sample_df_sent.sort_values(by=['timeStamp'])
    sample_df_sent['timeStamp'] = sample_df_sent['timeStamp'].astype('int')

    # Filtering of sent normal transfers to contract addresses
    sample_df_sent_contracts = sample_df[sample_df['contractAddress'] != '']

    # Compilation of normal sent transaction statistics
    core_stats_Sent_tnx = len(sample_df_sent)
    core_stats_MinValSent = sample_df_sent['eth value'].min()
    core_stats_MaxValSent = sample_df_sent['eth value'].max()
    core_stats_AvgValSent = sample_df_sent['eth value'].mean()
    core_stats_TotalEtherSent = sample_df_sent['eth value'].sum()
    core_stats_UniqueSentTo_Addresses = len(sample_df_sent['to'].unique())

    # Handling of received transactions stats
    sample_df_received =  sample_df[sample_df['txn type'] == 'received']
    sample_df_received = sample_df_received.sort_values(by=['timeStamp'])
    sample_df_received['timeStamp'] = sample_df_received['timeStamp'].astype('int')


    # Compilation of normal received transaction statistics
    core_stats_Received_tnx = len(sample_df_received)
    core_stats_MinValueReceived = sample_df_received['eth value'].min()
    core_stats_MaxValueReceived = sample_df_received['eth value'].max()
    core_stats_AvgValueReceived = sample_df_received['eth value'].mean()
    core_stats_TotalEtherReceived = sample_df_received['eth value'].sum()
    core_stats_UniqueReceivedFrom_Addresses = len(sample_df_received['from'].unique())

    # Handling of remaining normal transaction values
    sample_df['timeStamp'] = sample_df['timeStamp'].astype('int')
    sample_df.sort_values(by=['timeStamp'])
    sample_df['unix time difference'] = sample_df['timeStamp'].diff()
    sample_df_time_dim = sample_df.groupby('txn type')['unix time difference'].sum()/60


    # Compilation of remaining normal transaction statistics
    core_stats_TimeDiffbetweenfirstand_last = ((sample_df['timeStamp'].max()) - (sample_df['timeStamp'].min())) / 60
    core_stats_TotalTransactions = len(sample_df)
    core_stats_NumberofCreated_Contracts = len(sample_df[sample_df['contractAddress'] != ''])
    core_stats_Avg_min_between_received_tnx = sample_df_time_dim['received']/ core_stats_Received_tnx
    core_stats_Avg_min_between_sent_tnx = sample_df_time_dim['sent']/core_stats_Sent_tnx
    core_stats_TotalEtherBalance = core_stats_TotalEtherReceived - core_stats_TotalEtherSent
    compiled_normal_tnx_result = {'Address': address, 'FLAG': 1,
                                  'Avg min between sent tnx': core_stats_Avg_min_between_sent_tnx,
                                 'Avg min between received tnx': core_stats_Avg_min_between_received_tnx,
                                  'Time Diff between first and last (Mins)': core_stats_TimeDiffbetweenfirstand_last,
                                  'Unique Received From Addresses':core_stats_UniqueReceivedFrom_Addresses,
                                  'min value received': core_stats_MinValueReceived,
                                  'max value received ': core_stats_MaxValueReceived,
                                  'avg val received': core_stats_AvgValueReceived,
                                  'min val sent': core_stats_MinValSent,
                                  'avg val sent': core_stats_AvgValSent ,
                                  'total transactions (including tnx to create contract': core_stats_TotalTransactions,
                                  'total ether received': core_stats_TotalEtherReceived,
                                  'total ether balance':core_stats_TotalEtherBalance}
    return pd.DataFrame([compiled_normal_tnx_result])

def get_empty_details_for_address(address):
    compiled_empty_address = {
          'Address': address, 'FLAG': 1,
          'Avg min between sent tnx': 0,
         'Avg min between received tnx': 0,
          'Time Diff between first and last (Mins)': 0,
          'Unique Received From Addresses':0,
          'min value received': 0,
          'max value received ': 0,
          'avg val received': 0,
          'min val sent': 0,
          'avg val sent': 0 ,
          'total transactions (including tnx to create contract': 0,
          'total ether received': 0,
          'total ether balance':0
    }
    return pd.DataFrame([compiled_empty_address])


address_list = pd.read_csv('addresses_e_not_in_k_test.csv')

list_of_address = address_list['Address'].tolist()

# Initialize an empty DataFrame
base_df = pd.DataFrame()

# Process each address
for i, address in enumerate(list_of_address):
    try:
        cand_df = get_address_stats_normal_tnx(address)
        base_df = pd.concat([base_df, cand_df], ignore_index=True)
        print(f"Address number {i}: {address} processed.")
    except Exception as e:
        print(f"Error processing address {i}: {address}, {e}")
        cand_df = get_empty_details_for_address(address)
        base_df = pd.concat([base_df, cand_df], ignore_index=True)

# Final DataFrame is now `base_df`, ready for further analysis
print("DataFrame creation complete!")
print(base_df)


