using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using ImageConverter.Imaging;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public partial class MainForm : Form
	{
		private const string FileExtension = ".jpeg";

		private IConverter _converter;
		private ImageResizer _imageResizer;
		private ImageMaskCreator _maskCreator;

		public MainForm()
		{
			InitializeComponent();
			
			_imageResizer = new ImageResizer();
			_maskCreator = new ImageMaskCreator(16, 16);
		}

		private void ConvertFiles(string type)
		{
			if(type == "csv")
			{
				_converter = new TxtFileConverter(FileExtension);
			}
			else if(type == "xml")
			{
				_converter = new XmlFileConverter(FileExtension);
			}
			else throw new ArgumentException("Wrong converter type", nameof(type));

			string rootDir = _converter.RootImageFolderName;
			
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
					.Where(e => e.EndsWith(FileExtension) || e.EndsWith(".jpg") || e.EndsWith(".bmp"));


				foreach(var fullFileName in fullFileNames)
				{
					_converter.Convert(fullFileName);
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
			var rectangle = _maskCreator.GetRectangle(mask, img.Width, img.Height);

			img.Rectangle(new Rect(rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height), CvColor.OrangeRed, thickness: 2);
			using(var w = new Window(img))
			{
				Cv.WaitKey();
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
			ConvertFiles("csv");

			MessageBox.Show(this, $"Обработано {_converter.ImageCounter.ToString()} изображений", "Converted", MessageBoxButtons.OK);
		}

		private void OnButtonReadMaskClick(object sender, EventArgs e)
		{
			ReadMask();
		}

		private void OnButtonConvertFromXmlClick(object sender, EventArgs e)
		{
			ConvertFiles("xml");

			MessageBox.Show(this, $"Обработано {_converter.ImageCounter.ToString()} изображений", "Converted", MessageBoxButtons.OK);
		}

		private void button1_Click(object sender, EventArgs e)
		{
			var lines = File.ReadAllLines(@"E:\Study\Mallenom\fakes.txt", Encoding.Default);

			var trainData = File.ReadAllLines(@"E:\Study\Mallenom\train.txt", Encoding.Default);
			var trainDataList = new List<string>(trainData);

			foreach(var line in lines)
			{
				foreach(var item in trainData)
				{
					if(item.Contains(line) && line != "")
					{
						trainDataList.Remove(item);
						break;
					}
				}
			}
			

			File.WriteAllLines(@"E:\Study\Mallenom\train_new.txt", trainDataList.ToArray(), Encoding.Default);
			
		}

		private void OnButtonReadMaskClick(object sender, EventArgs e)
		{
			ReadMask();
		}
	}
}
