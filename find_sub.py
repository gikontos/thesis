from loader_test import load_all_data

data_list, annotation_list = load_all_data([], tsv_file="net/datasets/SZ2_training.tsv")

for rec_idx, annotation in enumerate(annotation_list):
    print(f"Processing recording {rec_idx+1}/{len(annotation_list)}")
    seizure_events = annotation_list[rec_idx].events
    for start, end in seizure_events:
        duration = end-start
        if duration > 900:
            print(duration)