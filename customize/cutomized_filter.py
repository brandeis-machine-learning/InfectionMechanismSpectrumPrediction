import csv
import os

mers = "Middle East respiratory syndrome-related coronavirus"
sars = "Severe acute respiratory syndrome-related coronavirus"
sars2 = "Severe acute respiratory syndrome coronavirus 2"
nl63 = "Human coronavirus NL63"
ACE2s = [sars, sars2, nl63]


# This customized filter removes
# 1) spike-receptor activities between (mers and ACE2)/(sars, sars2, nl63 and DPP4)
# 2) infection relationships which relies on FALSE binding predictions as suggested in point 1)
def customized_filter():
    proper_binding = []
    with open(os.path.abspath('../data/prediction/prediction_IMSP_interacts.csv'), 'r') as PPI_read, \
            open(os.path.abspath('../data/prediction/customized_prediction_interacts.csv'), 'w') as PPI_write:
        count = 0
        PPI_csv = csv.reader(PPI_read, delimiter=',')
        for row in PPI_csv:
            host1 = row[2].split(' ', 1)[1]
            if str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '') \
                    .rsplit(',', 1)[1] == 'unlikely':
                to_write = str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '') \
                               .rsplit(',', 1)[0] + ',unlikely' + '\n'
                PPI_write.write(to_write)
            elif row[2].startswith('Spike'):
                if (host1 in ACE2s and row[3].startswith('DPP4')) or (host1 == mers and row[3].startswith('ACE2')):
                    count = count + 1
                else:
                    to_write = str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '') \
                                   .rsplit(',', 1)[0] + ',likely' + '\n'
                    PPI_write.write(to_write)
                    proper_binding.append((host1 + ' ' + row[3].split(' ', 1)[1]))

            elif row[3].startswith('Spike'):
                host1 = row[3].split(' ', 1)[1]
                if (host1 in ACE2s and row[2].startswith('DPP4')) or (host1 == mers and row[3].startswith('ACE2')):
                    count = count + 1
                else:
                    to_write = str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '') \
                                   .rsplit(',', 1)[0] + ',likely' + '\n'
                    PPI_write.write(to_write)
                    proper_binding.append((host1 + ' ' + row[3].split(' ', 1)[1]))
            else:
                to_write = str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '') \
                               .rsplit(',', 1)[0] + ',likely' + '\n'
                PPI_write.write(to_write)
        PPI_write.close()

    print("# of removed improper bindings:", count)

    count = 0
    with open(os.path.abspath('../data/prediction/prediction_IMSP_infects.csv'), 'r') as infection_read, \
            open(os.path.abspath('../data/prediction/customized_prediction_infects.csv'), 'w') as infection_write:
        csv_reader = csv.reader(infection_read, delimiter=',')
        for row in csv_reader:
            virus = row[2].split(' ', 1)[1]
            host = row[3].split(' ', 1)[1]
            token = virus + ' ' + host
            if proper_binding.__contains__(token):
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',likely' + '\n'
                infection_write.write(to_write)
            else:
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',unlikely' + '\n'
                infection_write.write(to_write)
                count = count + 1
        infection_write.close()

    print("# of newly tagged unlikely infections:", count)


if __name__ == '__main__':
    customized_filter()
