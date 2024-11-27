import json
import tarfile


input_data = [

	{"inputs":"Tesla Stock Bounce Back After Earnings"
	},
	{"inputs":"Asthma is no longer a problem"
	},
	{"inputs":"AI is about to reach AGI according to top tech companies"
	},
	{"inputs":"New Lord of the Rings Animation movie widely acclaimed"
	},

]


def create_json_files(data):
	for i, d in enumerate(data):
		filename = f'input{i+1}.json'
		with open(filename,'w') as f:
			json.dump(d, f, indent=4)
			

def create_tar_file(input_files, output_filename='inputs.tar.gz'):
	with tarfile.open(output_filename, "w:gz") as tar:
		for file in input_files:
			tar.add(file)

def main():
	create_json_files(input_data)
	input_files = [f'input{i+1}.json' for i in range(len(input_data))]
	create_tar_file(input_files)


if __name__ == "__main__":
	main()
