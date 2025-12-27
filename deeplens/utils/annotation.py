import os
import csv


class Annotator():
    def __init__(self, root_dir: str, out_dir: str = None, header_list: list = None):
        self.root_dir = root_dir
        self.out_dir = (
            out_dir if out_dir
                    and out_dir.endswith('.csv') 
                    else os.path.join(out_dir or os.getcwd(), 'annotations.csv')
        )
        self.header_list = header_list or ['File', 'Label']
        self.label_map = {}
        self.root_dir_base = os.path.basename(root_dir)

    def makefile(self):
        try:
            for root, dirs, files in os.walk(self.root_dir):
                label = os.path.basename(root)
                if label != self.root_dir_base and label not in self.label_map:
                    self.label_map[label] = len(self.label_map)

            with open(self.out_dir, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header_list)
                for root, dirs, files in os.walk(self.root_dir):
                    label = os.path.basename(root)
                    for filename in files:
                        if filename.endswith('.wav'):
                            writer.writerow([filename, self.label_map[label]])
        except Exception as e:
            print(f'An error ocurred: {e}')

    
    def save_label_map(self, filepath='label_map.csv'):
        try:
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                for label, idx in self.label_map.items():
                    writer.writerow([label, idx])
        except Exception as e:
            print(f'Error saving label map: {e}')
