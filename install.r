options(repos = c(CRAN = "https://packagemanager.posit.co/cran/latest"))
# Now this will download pre-compiled packages (binaries)
install.packages(c("tidyverse", "arules", "arulesViz"))