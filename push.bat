echo "Start submitting code to the local repository"
echo "The current directory is：%cd%"
git add *
echo;
 
echo "Commit the changes to the local repository"
set now=%date% %time%
echo %now%
git commit -m "%now%"
echo;
 
echo "Commit the changes to the remote git server"
git push origin main
echo;
 
echo "Batch execution complete!"
echo;
