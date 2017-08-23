using System;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

using ImageConverter.Imaging;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public partial class MainForm : Form
	{
		private const string FileExtension = ".jpg";

		private CsvFileConverter _csvFileConverter;
		private ImageResizer _imageResizer;
		private ImageMaskCreator _maskCreator;

		public MainForm()
		{
			InitializeComponent();
			_csvFileConverter = new CsvFileConverter(FileExtension);
			_imageResizer = new ImageResizer();
			_maskCreator = new ImageMaskCreator(16, 16);
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

		private void ReadMask()
		{
			var filePath = string.Empty;
			var filter = "Text Files(*.txt;*.csv;)|*.txt;*.csv;";

			using(var fileDialog = new OpenFileDialog())
			{
				fileDialog.Filter = filter;
				if(fileDialog.ShowDialog(this) == DialogResult.OK)
				{
					filePath = fileDialog.FileName;
				}
			}

			var mask = _maskCreator.GetRegionsByMask(filePath, 8);
			_maskCreator.Mask = mask;

			//var imagePath = @"E:\Study\Mallenom\NeuralNetworks.TF\ImageConverter\ImageConverter\bin\Debug\images\s01\01.jpg";
			var imagePath = @"E:\Study\Mallenom\test.jpg";

			//Рисование изображения
			var img = new Mat(imagePath, LoadMode.Color);
			img = img.Resize(new OpenCvSharp.CPlusPlus.Size(ImageMaskCreator.Width, ImageMaskCreator.Heigth), 1, 1, Interpolation.Cubic);
			var rectangle = _maskCreator.GetRectangle(mask);

			img.Rectangle(new Rect(rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height), CvColor.Red);
			using(var w = new Window(img))
			{
				Cv.WaitKey();
			}
		}

		private void OnButtonConvertClick(object sender, EventArgs e)
		{
			ConvertFiles();

			MessageBox.Show(this, $"Обработано {_csvFileConverter.ImageCounter.ToString()} изображений", "Converted", MessageBoxButtons.OK);
		}

		private void OnButtonReadMaskClick(object sender, EventArgs e)
		{
			ReadMask();
		}
	}
}
