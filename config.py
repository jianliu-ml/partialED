bert_dir = 'data/BertModel/bert-base-cased/'

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)

event_set = ['Business_Merge_Org', 'Justice_Pardon', 'Personnel_End_Position', 'Transaction_Transfer_Money', 'Movement_Transport', 'Justice_Sue', 'Life_Divorce', 'Justice_Extradite', 'Personnel_Start_Position', 'Transaction_Transfer_Ownership', 'Business_Start_Org', 'Justice_Trial_Hearing', 'Personnel_Elect', 'Conflict_Attack', 'Justice_Execute', 'Justice_Charge_Indict', 'Contact_Meet', 'Justice_Fine', 'Conflict_Demonstrate', 'Life_Marry', 'Business_End_Org', 'Justice_Arrest_Jail', 'Justice_Convict', 'Personnel_Nominate', 'Life_Be_Born', 'Justice_Release_Parole', 'Justice_Acquit', 'Life_Injure', 'Contact_Phone_Write', 'Justice_Sentence', 'Business_Declare_Bankruptcy', 'Life_Die', 'Justice_Appeal']

bio_event_set = ['O'] + event_set
tag2idx = {tag: idx for idx, tag in enumerate(bio_event_set)}
idx2tag = {idx: tag for idx, tag in enumerate(bio_event_set)}