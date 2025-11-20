# Check if packages are installed, and install them if they are not
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse")
}
if (!requireNamespace("arules", quietly = TRUE)) {
    install.packages("arules")
}
if (!requireNamespace("arulesViz", quietly = TRUE)) {
    install.packages("arulesViz")
}