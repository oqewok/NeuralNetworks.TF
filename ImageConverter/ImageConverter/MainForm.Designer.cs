namespace ImageConverter
{
	partial class MainForm
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if(disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this._btnConvert = new System.Windows.Forms.Button();
			this._btnReadMask = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// _btnConvert
			// 
			this._btnConvert.Location = new System.Drawing.Point(12, 199);
			this._btnConvert.Name = "_btnConvert";
			this._btnConvert.Size = new System.Drawing.Size(75, 23);
			this._btnConvert.TabIndex = 0;
			this._btnConvert.Text = "Convert";
			this._btnConvert.UseVisualStyleBackColor = true;
			this._btnConvert.Click += new System.EventHandler(this.OnButtonConvertClick);
			// 
			// _btnReadMask
			// 
			this._btnReadMask.Location = new System.Drawing.Point(93, 199);
			this._btnReadMask.Name = "_btnReadMask";
			this._btnReadMask.Size = new System.Drawing.Size(75, 23);
			this._btnReadMask.TabIndex = 1;
			this._btnReadMask.Text = "Read mask";
			this._btnReadMask.UseVisualStyleBackColor = true;
			this._btnReadMask.Click += new System.EventHandler(this.OnButtonReadMaskClick);
			// 
			// MainForm
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(299, 241);
			this.Controls.Add(this._btnReadMask);
			this.Controls.Add(this._btnConvert);
			this.Name = "MainForm";
			this.Text = "Converter";
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Button _btnConvert;
		private System.Windows.Forms.Button _btnReadMask;
	}
}

