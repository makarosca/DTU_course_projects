#!/usr/bin/env python3
'''
Res genes: Dict, key = kmer, value = positions CHECK
Then compare kmers against it (first and last of the read)
Two files need to work at a time!
Create reverse complementary strands of reads!

To possibly improve:
in read_filename() make it possible for as many files to be added as we want
'''
import sys
import gzip

def read_filename():
	# read the filename
	if len(sys.argv) == 3:
		f1 = sys.argv[-2]
		f2 = sys.argv[-1]
	else:
		print("You can also run the script this way: python ngs_process.py <filename1> <filename2>")
		f1 = input('Enter <filename>: ')
		f2 = input('Enter another <filename>: ')
	filenames = [f1, f2]
	return filenames
	
	
def get_ngsread_kmer_list(dna, k):
	### returns a LIST of all k-mers extracted from an NGS read in the 
	### respective order

	ngsread_kmer = []
	for i in range(len(dna)-k+1):				# len-k ignores \n at the end
		ngsread_kmer.append(dna[i:i+k])
	return ngsread_kmer
	
def get_genes(filename):
	with open(filename, 'r') as infile:
		genes = []
		gene_names = []
		dna = ''
		for line in infile:
			if line[0] == '>':
				gene_names.append(line[:-1])
				if dna != '':
					genes.append(dna)
				dna = ''
			else:
				dna += line[:-1]
		genes.append(dna)
	return genes, gene_names


def complementary_strand(read):
	### returns a reverse complementary version of the read
	transTable = str.maketrans('ATCG', 'TAGC')
	complementDNA = read.translate(transTable)
	rev_dna = complementDNA[::-1]  
	
	return rev_dna

#a = read_filename()	#delete
#print(a[1])			#delete

def get_gene_kmer(genes, k):
	### returns a set of all k-mers taken from all resistance genes
	gene_kmer_dict = dict()
	
	for gene_num in range(len(genes)):
		for i in range( len(genes[gene_num]) -k+1):		# len-k+1 takes the last symbol too
			check = gene_kmer_dict.get(genes[gene_num][i:i+k])
			gene_kmer_dict[genes[gene_num][i:i+k]] = {gene_num: i}	# value = kmer position [which gene][where in the gene]

	return gene_kmer_dict


def align_ngs_gene(read, gene, start_check, end_check, gene_num, k):
	### returns the region of the gene where the read was successfully aligned, otherwise returns None

	SNP = 0
	update_reg = None

	if start_check is not None and start_check.get(gene_num) is not None:
		if end_check is not None and end_check.get(gene_num) is not None:
			# end_check[gene_num] is the position of beginning of the last k-mer
			# hence 'end_check[gene_num]+k' is the next nucleotide after the gene part
			part_of_gene = gene[ start_check[gene_num]:end_check[gene_num]+k ]

			# if both first and last k-mer of the read ARE present in the gene, align the whole read
			if read == part_of_gene:
				update_reg = [start_check[gene_num], end_check[gene_num]+k]
			
			# if the read is not identical to the part of gene, allow one SNP difference
			elif len(read) == len(part_of_gene):
				i = 0 			# number of the k-mer (position in the read)
				pos = 0 		# position in the gene sequence
				while i < len(read_kmer) and pos < len(gene):
					pos = start_check[gene_num] + i
					if read_kmer[i] != gene[pos:pos+k]:
						SNP += 1
						i += k
					else:
						i += 1
				if SNP == 1:
					update_reg = [start_check[gene_num], end_check[gene_num]+k]
				
				# discard if not identical and SNP > 1
				else:
					update_reg = None
			else:
				update_reg = None

		# if only the first k-mer of the read is found, align to the end of gene			
		else:
			gene_end = gene[-k:]
			end_pos = read.find(gene_end)
			part_of_gene = gene[ start_check[gene_num]:]
			if end_pos != -1 and read[:end_pos+k] == part_of_gene:
				update_reg = [start_check[gene_num], len(gene)]
			else:
				update_reg = None

	# if only the last k-mer of the read is found, align to the beginning of gene
	else:
		if end_check is not None and end_check.get(gene_num) is not None:
			gene_start = gene[:k]
			start_pos = read.find(gene_start)
			part_of_gene = gene[:end_check[gene_num]+k]
			if start_pos != -1 and read[start_pos:] == part_of_gene:
				update_reg = [0, end_check[gene_num]+k]
			else:
				update_reg = None
		else:
			update_reg = None

	return update_reg


if __name__ == "__main__":

	k = 19 #setting the k-mer length
	#NGS data files
	filenames = read_filename()

	#filenames = ["Unknown3_raw_reads_1.txt.gz", "Unknown3_raw_reads_2.txt.gz"]

	###processing resistance genes
	#Obtaining lists of resistance gene sequences and names
	[gene_list, gene_names] = get_genes('resistance_genes.fsa')

	#Obtaining a dict of res_gene k-mers with their respective locations as values
	# Dict of dicts: gene_dict [k-mer sequence] [gene_num] == position of the k-mer in the specific gene
	gene_dict = get_gene_kmer(gene_list, k)

	###creating a template for output
	#Creating depth list
	depth = []
	for gene_num in range(len(gene_list)):
		d1 = [0 for j in range(len(gene_list[gene_num])) ]
		depth.append(d1)


	###processing the reads

	#opening both read files
	read_count = 0
	for file in filenames:		#to iterate through both files
		with gzip.open(file, "r") as infile:
			for b_line in infile:
				line = b_line.decode('ASCII')

				if line[0] == '@':				# getting the DNA
					flag = True					# sequence through
				elif flag:						# stateful parsing
					if line[0] == '+':
						flag = False
					else:
						#read_count += 1
						dna_read = line[:-1]

						#get the reverse complement for the read
						both_reads = [dna_read, complementary_strand(dna_read)]
						read_count += 1

						# printing percentage of the reads processed
						if read_count > 0 and read_count % 100000 == 0:
							print(f"{read_count / 6938342 * 100:.2f}%")

						for read in both_reads:			# iterate through both strands
							#create kmer lists for both
							read_kmer = get_ngsread_kmer_list(read, k)

							#checking if extremities of the read fit the dict (initial read elimination)
							start_check = gene_dict.get(read_kmer[0])			# contain positions of the k-mers
							end_check = gene_dict.get(read_kmer[-1])			# in every relevant gene

							# set of gene numbers (gene id) to which the read will be aligned
							genes_to_analyze = set()
							if start_check is not None:
								for hit in start_check.keys():
									genes_to_analyze.add(hit)
							if end_check is not None:
								for hit in end_check.keys():
									genes_to_analyze.add(hit)

							# iterate through every relevant gene
							for gene_num in genes_to_analyze:
								# region that has to be updated
								update_reg = align_ngs_gene(read, gene_list[gene_num],
									start_check, end_check, gene_num, k)

								# if the read is not discarded, increase the depth of
								# gene by 1 in relevant region
								if update_reg is not None:
									for i in range(update_reg[0], update_reg[1]):
										depth[gene_num][i] += 1

	# calculate coverage and average
	coverageCount = 0 			# number of genes with coverage > 95%
	coverage = dict()			# coverage of each gene
	avg_depth = []
	for gene_num in range(len(depth)):
		hitCount = 0
		avg_depth.append(0)
		for i in range(len(depth[gene_num])):
			if depth[gene_num][i] >= 10:
				hitCount += 1
			avg_depth[gene_num] += depth[gene_num][i]
		coverage[gene_num] = hitCount/len(depth[gene_num])
		avg_depth[gene_num] = avg_depth[gene_num]/len(depth[gene_num])
		if coverage[gene_num] > 0.95:
			coverageCount += 1

	sorted_gene_num = sorted(coverage.keys(), key=coverage.get, reverse=True)
	print(coverageCount, " genes have achieved coverage above 95%:\n", sep='')
	i = 0
	for gene_num in sorted_gene_num[:coverageCount]:
		i += 1
		# display the gene names that are covered, their coverage and average depth
		print(i, '. ', gene_names[gene_num], sep='');
		print("Gene coverage: ", "{:.2f}".format(coverage[gene_num]*100), "%; Gene depth: ",
			"{:.2f}".format(avg_depth[gene_num]), '\n', sep='')
		#print(depth[gene_num])						# To see corresponding depth distribution
