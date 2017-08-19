setwd("~/odrive/LBL/Vortex Beam/Figure Data/Data")
library(ggplot2)
library(xlsx)
library(reshape2)
library(viridis)
library(scales)
library(ggpubr)

# Settings
# ----------------------------------------------------------------------------------------------------------------------------------------- #
sample = 'SS'
pulses = c('gp', 'm1a', 'm1r', 'm2')
titles = c('Gaussian', 'M1, Azimuthal', 'M1, Radial', 'M2')
wavelength_range = c(370, 395)  # (nm)
delay_max =  1500 # (ns)
bkg = 5.33  # (a.u.) used to calculate SNR
# ----------------------------------------------------------------------------------------------------------------------------------------- #

# Create subplot for pulse type
pers_plot = function(sample, pulse, title, w=wavelength_range, dm=delay_max, sbr=sbr_lim) {
  # Read data
  raw_d = read.xlsx(paste(sample, pulse, 'Persistence.xlsx'), sheetIndex=1, check.names=FALSE)
  d = melt(raw_d, id=c('wavelength'))
  
  # Remove negative values, calculate bkr
  d$value[d$value <= 0] = 1
  d$sbr = d$value / bkg
  
  # Subset to selected wavelength range
  d = d[(d$wavelength > w[1]) & (d$wavelength < w[2]),]

  # Subset to selected delay time range
  d$delay = as.numeric(d$variable) * 50
  d = d[d$delay < dm, ]
  
  # Set scalebar ticks
  mybreaks=c(3, 10, 100, 1000, 10000)
  low = mybreaks[1]
  high = mybreaks[length(mybreaks)]
  
  # Create plot
  p = ggplot(d, aes(x=d$wavelength, y=d$delay))
  plot = p + geom_raster(aes(fill=ifelse(sbr < 3, 2, sbr)), interpolate=T) + 
    scale_fill_viridis(option='inferno', trans='log', limits=c(low, high), breaks=mybreaks, labels=mybreaks, oob=squish) + 
    labs(x='Wavelength (nm)', y='Delay Time (ns)', fill='Signal to\nNoise Ratio\n', title=title)
  return(plot)
}

# Create subplots for each pulse type
fig = list()
for (i in 1:4) {
  fig[[i]] = pers_plot(sample, pulses[i], titles[i])
}

# Create figure of subplots
fig[[1]]  # just Gaussian
ggarrange(fig[[1]], fig[[2]], fig[[3]], fig[[4]], ncol=2, nrow=2)  # all
