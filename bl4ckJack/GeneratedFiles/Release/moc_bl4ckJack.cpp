/****************************************************************************
** Meta object code from reading C++ file 'bl4ckJack.h'
**
** Created: Wed Jan 5 23:30:11 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../bl4ckJack.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'bl4ckJack.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_bl4ckJack[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      15,   11,   10,   10, 0x0a,
      65,   62,   10,   10, 0x0a,
     100,   62,   10,   10, 0x0a,
     142,  140,   10,   10, 0x0a,
     186,  179,   10,   10, 0x08,
     235,   10,   10,   10, 0x08,
     252,   10,   10,   10, 0x08,
     269,   10,   10,   10, 0x08,
     277,   10,   10,   10, 0x08,
     284,   10,   10,   10, 0x08,
     292,   10,   10,   10, 0x08,
     313,   10,   10,   10, 0x08,
     334,   10,   10,   10, 0x08,
     356,   10,   10,   10, 0x08,
     373,   10,   10,   10, 0x08,
     402,   10,   10,   10, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_bl4ckJack[] = {
    "bl4ckJack\0\0,,,\0"
    "updateUIFileAdd(QString,QString,QString,float)\0"
    ",,\0updateBruteStatus(int,int,QString)\0"
    "updateBruteLabels(double,qint64,qint64)\0"
    ",\0updateBrutePassword(QString,QString)\0"
    "reason\0iconActivated(QSystemTrayIcon::ActivationReason)\0"
    "messageClicked()\0showProperties()\0"
    "start()\0stop()\0pause()\0hashTableInputHash()\0"
    "hashTableInputFile()\0hashTableDeleteHash()\0"
    "hashTableClear()\0PasswordSaveFileTableClear()\0"
    "PasswordSaveFileTable()\0"
};

const QMetaObject bl4ckJack::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_bl4ckJack,
      qt_meta_data_bl4ckJack, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &bl4ckJack::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *bl4ckJack::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *bl4ckJack::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_bl4ckJack))
        return static_cast<void*>(const_cast< bl4ckJack*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int bl4ckJack::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateUIFileAdd((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 1: updateBruteStatus((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 2: updateBruteLabels((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< qint64(*)>(_a[2])),(*reinterpret_cast< qint64(*)>(_a[3]))); break;
        case 3: updateBrutePassword((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 4: iconActivated((*reinterpret_cast< QSystemTrayIcon::ActivationReason(*)>(_a[1]))); break;
        case 5: messageClicked(); break;
        case 6: showProperties(); break;
        case 7: start(); break;
        case 8: stop(); break;
        case 9: pause(); break;
        case 10: hashTableInputHash(); break;
        case 11: hashTableInputFile(); break;
        case 12: hashTableDeleteHash(); break;
        case 13: hashTableClear(); break;
        case 14: PasswordSaveFileTableClear(); break;
        case 15: PasswordSaveFileTable(); break;
        default: ;
        }
        _id -= 16;
    }
    return _id;
}
static const uint qt_meta_data_InputHashWorker[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      21,   17,   16,   16, 0x05,

 // slots: signature, parameters, type, tag, flags
      68,   16,   16,   16, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_InputHashWorker[] = {
    "InputHashWorker\0\0,,,\0"
    "updateUIFileAdd(QString,QString,QString,float)\0"
    "doWork()\0"
};

const QMetaObject InputHashWorker::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_InputHashWorker,
      qt_meta_data_InputHashWorker, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &InputHashWorker::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *InputHashWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *InputHashWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_InputHashWorker))
        return static_cast<void*>(const_cast< InputHashWorker*>(this));
    return QThread::qt_metacast(_clname);
}

int InputHashWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateUIFileAdd((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 1: doWork(); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void InputHashWorker::updateUIFileAdd(QString _t1, QString _t2, QString _t3, float _t4)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
