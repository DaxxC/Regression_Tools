def split_chars(text):
  return " ".join(list(text))

def get_lines(filename):
  """
  Reads text filename and returns the lines of text as a list.
  Args: 
    filename: a str containing the target filepath.

  Returns: 
    A list of strings with one string per line from the target filename.
  """
  with open(filename, 'r') as f:
    return f.readlines()
    
def preprocess_text_with_line_numbers(filename):
  """ Returns a list of dictionaries of abstract line data. 
  Takes in filename, reads its contents and sorts through each line,
   extracts things like the target label, the text of the sentence, how many sentences are in the current
   abstract and what sentence number the target line is.
   Used on file formats like pubmed-rct/PubMed_200k_RCT files.
  """
  input_lines=get_lines(filename)
  abstract_lines=''
  abstract_samples=[]

  for line in input_lines:
    if line.startswith('###'):
      abstract_id=line
      abstract_lines=''

    elif line.isspace():
      abstract_line_split=abstract_lines.splitlines()

      for abstract_line_number, abstract_line in enumerate (abstract_line_split):
        line_data= {}
        target_text_split= abstract_line.split('\t')
        line_data['target']=target_text_split[0]
        line_data['text']=target_text_split[1].lower()
        line_data['line_number']=abstract_line_number
        line_data['total_lines']=len(abstract_line_split)-1
        abstract_samples.append(line_data)
    
    else:
      abstract_lines+=line

  return abstract_samples
