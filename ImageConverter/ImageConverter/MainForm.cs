using System;
using System.Data;
using System.IO;
using System.Linq;
using System.Windows.Forms;

using ImageConverter.Imaging;


namespace ImageConverter
{
	public partial class MainForm : Form
	{
		private const string FileExtension = ".jpg";

		private CsvFileConverter _csvFileConverter;
		private ImageResizer _imageResizer;

		public MainForm()
		{
			InitializeComponent();
			_csvFileConverter = new CsvFileConverter(FileExtension);
			_imageResizer = new ImageResizer();
		}

		private void ConvertFiles()
		{
			var rootDir = Path.Combine(Directory.GetCurrentDirectory(), CsvFileConverter.RootImageFolderName);

			if(!Directory.Exists(rootDir))
			{
				Directory.CreateDirectory(rootDir);
			}

			Convert(rootDir);
		}

		private void Convert(string path)
		{
			var directories = Directory.EnumerateDirectories(path);

			if(directories.Count() != 0)
			{
				foreach(var directory in directories)
				{
					Convert(directory);
				}
			}
			else
			{
				var fullFileNames = Directory
					.EnumerateFiles(path)
					.Where(e => e.EndsWith(FileExtension));


				foreach(var fullFileName in fullFileNames)
				{
					_csvFileConverter.Convert(fullFileName);
				}
			}
		}

		private void OnButtonConvertClick(object sender, EventArgs e)
		{
			ConvertFiles();

			MessageBox.Show(this, $"Обработано {_csvFileConverter.ImageCounter.ToString()} изображений", "Converted", MessageBoxButtons.OK);
		}
	}
}
