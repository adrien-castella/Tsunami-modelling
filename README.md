# Tsunami-modelling

Instructions to running new_modelling.py

Go to code location in command prompt using "cd C:\" + PATH

1. Run the command "py new_modelling.py" or "python new_modelling.py"
2. Answer the questions:
	- First comes the scale. Best options are 0.5, 0.75, and 1
		* the scale indicated how "pixaleted" the simulation is
		* in other words the size of a pixel
		* a pixel represents 5/scale km x 5/scale km
	- Second comes the timesteps. Suggestion is 1000. The number of
	  timesteps is self explanatory
	- Third comes the amplitude. This is the highest the wave is at initiation
		* This value should be given in meters
	- Fourth comes the name of the files. Self-explanatory
	- Fifth comes the timestep size. A suggested max will be given above
		* The max indicates the maximal timestep to ensure stability
	- Sixth comes the radius of the wave. It is suggested to choose it
	  between 36 and 90. Determine it visually using the image
	- Seventh comes the wavelength. Suggested to choose 16, 18, 20, or 22. Hard
	  to interpret in physical units.
	- Eight comes the number of waves. The number should always be odd. The # of
	  waves is given by (x-1)/2 where x is the number you provided.
3. Interpreting the results:
	- Two images of the first two frames of the wave
	- Two JSON files
		* _viz is for people to read (includes the chosen parameters)
		* _py is for the next step
	- Topography PNG simply names "topography.png"
	- Tsunami simulation video named "Tsunami simulation "+name+".mp4"
4. Plotting the amplitude near the narrow channel
	- In the same command prompt as step 1, run the command "py plot_json.py"
	  or "python plot_json.py"
	- Answer the questions
		* enter the same name as in step 2