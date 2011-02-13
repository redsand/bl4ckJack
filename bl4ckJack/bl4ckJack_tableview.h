 #ifndef TABLEMODEL_H
 #define TABLEMODEL_H

 #include <QAbstractTableModel>
 #include <QPair>
 #include <QList>
 #include <QItemSelection>

 class QSortFilterProxyModel;
 class QItemSelectionModel;

//! TableModel Class
/**
 * TableModel Class
 * TableModel Class used for quickly viewing and managing a large list of hashes.
 */
 class TableModel : public QAbstractTableModel
 {
     Q_OBJECT

 public:
	//! TableModel Constructor
	/**
	  * TableModel Constructor
	  * Used for quickly viewing and managing a large list of hashes.
	  * @param QObject *parent
      * @see TableModel()
      * @see ~TableModel()
      * @return None
	  */
     TableModel(QObject *parent=0);
	 
	//! TableModel Constructor
	/**
	  * TableModel Constructor
	  * Used for quickly viewing and managing a large list of hashes.
	  * @param QList< QPair<QString, QString> >
	  * @param QObject *parent
      * @see TableModel()
      * @see ~TableModel()
      * @return None
	  */
     TableModel(QList< QPair<QString, QString> > listofPairs, QObject *parent=0);
	 
	//! TableModel Destructor
	/**
	  * TableModel Destructor
	  * Used for quickly viewing and managing a large list of hashes.
	  * @see TableModel()
      * @return None
	  */
	 ~TableModel()
	 {
	 }

	//! TableModel Get Row Count
	/**
	  * TableModel Get Row Count
	  * return the number of rows in the table.
	  * @param const QModelIndex &
	  * @see TableModel()
      * @return int
	  */
     int rowCount(const QModelIndex &parent) const;
	 
	 
	//! TableModel Get Column Count
	/**
	  * TableModel Get Column Count
	  * return the number of rows in the table.
	  * @param const QModelIndex &
	  * @see TableModel()
      * @return int
	  */
     int columnCount(const QModelIndex &parent) const;
	 
	//! TableModel data
	/**
	  * TableModel data
	  * return the data supplied by index and role
	  * @param const QModelIndex &
	  * @param int role
	  * @see QVariant
      * @return QVariant
	  */
     QVariant data(const QModelIndex &index, int role) const;
	 
	//! TableModel headerData
	/**
	  * TableModel headerData
	  * return the header data supplied by section, orientation and role
	  * @param int section
	  * @param Qt::Orientation
	  * @param int role
	  * @see QVariant
      * @return QVariant
	  */
     QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	 
	//! TableModel flags
	/**
	  * TableModel flags
	  * return the item's flags state
	  * @param const QModelIndex & index
	  * @see Qt::ItemFlags
      * @return Qt::ItemFlags
	  */
     Qt::ItemFlags flags(const QModelIndex &index) const;
	 
	//! TableModel setData
	/**
	  * TableModel setData
	  * set the data specified by the index, value and role
	  * @param const QModelIndex & index
	  * @param const QVariant & value
	  * @param int role (QT::EditRole)
	  * @see Qt::ItemFlags
      * @return bool
	  */
     bool setData(const QModelIndex &index, const QVariant &value, int role=Qt::EditRole);
	 
	//! TableModel insertRows
	/**
	  * TableModel insertRows
	  * insert a row based off of position and row count
	  * @param int position
	  * @param int rows
	  * @param const QModelIndex & index
	  * @see QModelIndex
      * @return bool
	  */
     bool insertRows(int position, int rows, const QModelIndex &index=QModelIndex());
	 
    //! TableModel removeRows
	/**
	  * TableModel removeRows
	  * remove a row based off of position and row count
	  * @param int position
	  * @param int rows
	  * @param const QModelIndex & index
	  * @see QModelIndex
      * @return bool
	  */
 	 bool removeRows(int position, int rows, const QModelIndex &index=QModelIndex());
	 
	//! TableModel Get QT List
	/**
	  * TableModel Get QT List
	  * return a QT List of pairs from the given model
	  * @see QList
	  * @see QPair
      * @return QList< QPair<QString, QString> >
	  */
     QList< QPair<QString, QString> > getList();
	 
	//! TableModel Read Contents From File
	/**
	  * TableModel Read Contents From File
	  * read the contents of a file and display it in the model view
	  * @param QString filename
      * @return void;
	  */
	 void readFromFile(QString fileName);
	 
	//! TableModel Write Contents From File
	/**
	  * TableModel Write Contents From File
	  * write the contents of a file from the model view that is being displayed.
	  * @param QString filename
      * @return void;
	  */
     void writeToFile(QString fileName);

 public slots:
 
	//! TableModel Add Entry To ModelView
	/**
	  * TableModel Add Entry To ModelView
	  * add entries to our model view utilizing QT slots and messages.
	  * @param QString name
	  * @param QString address
      * @return void;
	  */
     void addEntry(QString name, QString address);

 signals:
	//! TableModel Detect Selection Changed
	/**
	  * TableModel Detect Selection Changed
	  * Detect if our select has changed and react based upon it.
	  * @param const QItemSelection & selected
      * @return void;
	  */
     void selectionChanged (const QItemSelection &selected);

 private:
	//! Internal list of pairs
     QList< QPair<QString, QString> > listOfPairs;
	 
	//! Internal filter for sorting our list based upon our requirements.
     QSortFilterProxyModel *proxyModel;
 };



 #endif