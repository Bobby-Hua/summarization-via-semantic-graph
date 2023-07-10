## AMR Graph Preprocessing
### External Repository Needed:

* AMR-gs (https://github.com/jcyk/AMR-gs)
* wl-coref (https://github.com/vdobrovolskii/wl-coref)
* JAMR (Linux and macOS) (https://github.com/jflanigan/jamr)
    
Please refer to the readme from their repository for installation instructions, you may want to install them in separate conda environments.

### Preprocessing steps:
1.  Preprocess the text file:
	-  Preprocessing varies for different datasets, we recommend split sentences that are too long ( >400 tokens), as they tend to be problematic for AMR-gs.
	-   Use C99 and sentence BERT to segment text into scenes segments, we recommend the result scenes have length < 600 tokens.
	-   You will need the text in json format, and also a separate json file to keep track of the number of scenes in each episode/meeting.
	-   Check sample files for reference.

2.  Parse all sentences into sentence-level AMR by AMR-gs:
	-   Refer to [https://github.com/jcyk/AMR-gs#amr-parsing-with-pretrained-models](https://github.com/jcyk/AMR-gs#amr-parsing-with-pretrained-models)

3.  Create scene-AMRs from text segments and AMR-gs output, and format for JAMR input:

	    python jamr_prep.py amr_gs_file input_text_file jamr_dir

	-   \* this will generate one file per scene in jamr_dir_in.
    
4.  Use JAMR Aligner to align scene-AMR graph with text:
	-   Refer to [https://github.com/jflanigan/jamr#running-the-aligner](https://github.com/jflanigan/jamr#running-the-aligner)
		-  We included a script for multi-threaded processing:
		
	           bash jamr_multi_processing number_of_scenes jamr_dir_in jamr_dir_out

5.  Using outputs from JAMR, obtain coreferences with wl-coref:
	-   Prepare inputs to wl-coref:
	
		    python coref_prep.py input_text_file jamr_dir_out coref_input_file
	
	-   Use wl-coref to get coreference relations:
		-   Refer to [https://github.com/vdobrovolskii/wl-coref#prediction](https://github.com/vdobrovolskii/wl-coref#prediction)
		
		        python predict.py roberta coref_input_file coref_output_file

6.  Put everything together! Use JAMR alignment output, wl-coref coreference output to create the final graph output:

		python final_prep.py input_text_file jamr_dir_out coref_output_file
